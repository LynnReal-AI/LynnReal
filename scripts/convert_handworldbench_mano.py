"""Fit HandWorldBench joint tracks to MANO and render LynnReal-Hand control videos."""

import argparse
import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


FINGERTIPS = [745, 317, 444, 556, 673]
JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MCP = [5, 9, 13, 17]


def load_case(dataset, case_id):
    with (dataset / "prompts.jsonl").open() as stream:
        case = next((c for c in map(json.loads, stream) if c["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"Unknown HandWorldBench case: {case_id}")
    with np.load(dataset / "hands" / f"{case_id}.npz") as hands:
        points = hands["joints_camera_m"]
        valid = hands["valid"]
        indices = hands["frame_index"]
    with np.load(dataset / "camera" / f"{case_id}.npz") as camera:
        intrinsic = camera["K"]
        distortion = camera["D"]
        fisheye = "fisheye" in str(camera["distortion_model"])
        if not np.array_equal(camera["frame_index"], indices):
            raise ValueError("Hand and camera frame indices differ")
    count = case["frame_count"]
    if points.shape != (count, 2, 21, 3) or valid.shape != points.shape[:-1]:
        raise ValueError("Expected a (frames, 2, 21, 3) camera-space hand track with validity flags")
    if case["category"] not in {"human", "real_robot", "handson"} or case["fps"] <= 0:
        raise ValueError("Unsupported embodiment or invalid frame rate")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or np.any(intrinsic.diagonal()[:2] <= 0):
        raise ValueError("Invalid camera intrinsics")
    if not case["prompt"].strip() or not np.isfinite(distortion).all():
        raise ValueError("Empty action prompt or invalid lens calibration")
    return case, points, valid, intrinsic, distortion, fisheye


def load_mano(directory, side):
    # Original MANO pickle files contain objects from the legacy chumpy package.
    for name, builtin in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                          ("object", object), ("unicode", str), ("str", str)):
        if name not in np.__dict__:
            setattr(np, name, builtin)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    import smplx

    return smplx.MANO(model_path=str(directory), is_rhand=side == 1, use_pca=False,
                      flat_hand_mean=True).cuda().eval().requires_grad_(False)


def fit_hand(model, points, valid, human, steps):
    count = len(points)
    valid = valid & np.isfinite(points).all(axis=-1)
    anchors = [0] if human else MCP
    visible = valid[:, anchors].all(axis=1) & (valid.sum(axis=1) >= 8)
    if not visible.any():
        return None, visible
    clean = np.nan_to_num(points, nan=0, posinf=0, neginf=0).astype(np.float32)
    center = clean[:, anchors].mean(axis=1, keepdims=True)
    relative = clean - center
    target = torch.from_numpy(relative).cuda()
    weight = valid.astype(np.float32)
    weight[:, [4, 8, 12, 16, 20]] *= 2
    weight[:, MCP] *= 2
    if not human:
        weight[:, 0] = 0  # The robot palm center is not a human wrist joint.
    weight = torch.from_numpy(weight * visible[:, None]).cuda()
    pose = torch.zeros(count, 45, device="cuda", requires_grad=True)
    orientation = torch.zeros(count, 3, device="cuda", requires_grad=True)
    shape = torch.zeros(1, 10, device="cuda", requires_grad=True)
    log_scale = torch.zeros(1, device="cuda", requires_grad=True)

    def geometry():
        output = model(hand_pose=pose, global_orient=orientation, betas=shape.expand(count, -1))
        joints = torch.cat((output.joints, output.vertices[:, FINGERTIPS]), dim=1)[:, JOINT_ORDER]
        origin = joints[:, anchors].mean(dim=1, keepdim=True)
        return (output.vertices - origin) * log_scale.exp(), (joints - origin) * log_scale.exp()

    with torch.no_grad():
        _, joints = geometry()
        p, q = joints[:, MCP].cpu().numpy(), relative[:, MCP]
        ratios = np.linalg.norm(q, axis=(1, 2)) / np.linalg.norm(p, axis=(1, 2)).clip(1e-8)
        log_scale.fill_(float(np.log(np.clip(np.median(ratios[visible]), 0.25, 3))))
        u, _, vh = np.linalg.svd(np.einsum("bni,bnj->bij", p, q))
        for i in np.flatnonzero(visible):
            rotation = vh[i].T @ u[i].T
            if np.linalg.det(rotation) < 0:
                vh[i, -1] *= -1
                rotation = vh[i].T @ u[i].T
            orientation[i].copy_(torch.from_numpy(cv2.Rodrigues(rotation)[0][:, 0]))

    contiguous = torch.from_numpy(visible[:-2] & visible[1:-1] & visible[2:]).cuda()
    optimizer = torch.optim.Adam([pose, orientation, shape, log_scale], lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, joints = geometry()
        error = ((joints - target).square().sum(dim=-1) * weight).sum() / weight.sum().clamp_min(1)
        loss = error + 1e-7 * pose.square().mean() + 1e-8 * shape.square().mean()
        if contiguous.any():
            acceleration = pose[2:] - 2 * pose[1:-1] + pose[:-2]
            loss = loss + 1e-6 * acceleration[contiguous].square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite MANO fitting objective")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            shape.clamp_(-3, 3)
            log_scale.clamp_(np.log(0.25), np.log(3))
    with torch.no_grad():
        vertices, _ = geometry()
        return vertices.cpu().numpy() + center, visible


def rasterize(context, fitted, faces, frame, camera, size, source_size):
    import nvdiffrast.torch as dr

    width, height = size
    sx, sy = width / source_size[0], height / source_size[1]
    k, distortion, fisheye = camera
    project = cv2.fisheye.projectPoints if fisheye else cv2.projectPoints
    clips, triangles, normals, triangle_side = [], [], [], []
    offset = 0
    for side, (track, visible) in enumerate(fitted):
        if not visible[frame]:
            continue
        vertices = track[frame].astype(np.float64)
        uv = project(vertices.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
        z = vertices[:, 2]
        vertices[:, :2] = (uv - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]] * z[:, None]
        vertices = vertices.astype(np.float32)
        mesh = faces if side else faces[:, [0, 2, 1]]
        a, b, c = vertices[mesh].transpose(1, 0, 2)
        face_normals = np.cross(b - a, c - a)
        normal = np.zeros_like(vertices)
        for corners in mesh.T:
            np.add.at(normal, corners, face_normals)
        normals.append(normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8))

        u, v = ((uv + 0.5) * [sx, sy] - 0.5).T
        near, far = 0.01, 100.0
        depth = ((far + near) * z - 2 * far * near) / (far - near)
        clips.append(np.column_stack(((2*u/width - 1)*z, (1 - 2*v/height)*z, depth, z)))
        triangles.append(mesh + offset)
        triangle_side.extend([side] * len(mesh))
        offset += len(vertices)
    if not clips:
        return np.zeros((height, width, 3), np.uint8)
    coords = torch.from_numpy(np.concatenate(clips).astype(np.float32)).cuda()[None].contiguous()
    mesh = torch.from_numpy(np.concatenate(triangles).astype(np.int32)).cuda().contiguous()
    attributes = torch.from_numpy(np.concatenate(normals)).cuda()[None].contiguous()
    rast, _ = dr.rasterize(context, coords, mesh, resolution=[height, width])
    rast = rast.flip(1)
    interpolated, _ = dr.interpolate(attributes, rast, mesh)
    n = interpolated[0]
    cosine = (n[:, :, 2] / (n.norm(dim=-1) + 1e-8)).abs()
    tri = rast[0, :, :, 3].int()
    mask = tri > 0
    side = torch.as_tensor(triangle_side, device="cuda")[tri[mask] - 1]
    colors = torch.tensor([[0, 0, 255], [255, 0, 0]], device="cuda", dtype=torch.float32)  # RGB
    light = cosine[mask].unsqueeze(-1)
    rgb = torch.zeros(height, width, 3, device="cuda")
    rgb[mask] = colors[side] * (0.3 + 0.7 * light).clamp(0, 1) + 102 * light.pow(32)
    return rgb.clamp(0, 255).byte().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="HandWorldBench snapshot directory")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mano-dir", type=Path, help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl")
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--size", type=int, nargs=2, default=(1344, 768), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--steps", type=int, default=650, help="MANO optimization steps")
    parser.add_argument("--check-only", action="store_true", help="Validate source inputs without fitting")
    args = parser.parse_args()
    case, points, valid, k, distortion, fisheye = load_case(args.dataset, args.case_id)
    if args.check_only:
        print(f"{case['case_id']}: {len(points)} frames, {case['fps']} fps, {case['category']}")
        return
    if args.output is None or args.mano_dir is None:
        parser.error("conversion requires --mano-dir and --output")
    if args.steps <= 0 or any(n <= 0 or n % 2 for n in args.size):
        parser.error("use positive steps and positive even width and height")
    import nvdiffrast.torch as dr

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fitted = []
    for side in (0, 1):
        model = load_mano(args.mano_dir, side)
        fitted.append(fit_hand(model, points[:, side], valid[:, side], case["category"] == "human", args.steps))
    context = dr.RasterizeCudaContext()
    width, height = args.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        path = Path(temp) / "hand.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{width}x{height}", "-r", str(case["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder, torch.no_grad():
            for frame in range(len(points)):
                rgb = rasterize(context, fitted, model.faces, frame, (k, distortion, fisheye),
                                args.size, (case["width"], case["height"]))
                encoder.stdin.write(rgb.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not encode the hand control video")
        os.replace(path, args.output)


if __name__ == "__main__":
    main()
