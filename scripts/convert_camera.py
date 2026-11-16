import argparse
from lynnreal_hand.camera import convert_camera
p=argparse.ArgumentParser(description='Physical C2W poses to native H3 latent-aligned conditions')
p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,required=True)
a=p.parse_args()
print(convert_camera(a.input,a.output,frames=a.frames))
