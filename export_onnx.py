"""Export a trained checkpoint for FastAPI/ONNX Runtime inference."""
import argparse
import json
from pathlib import Path
import torch
from models import build_model


def export(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    arch = checkpoint['architecture']
    size = int(checkpoint['size'])
    model = build_model(arch, checkpoint['encoder'], pretrained=False).eval()
    model.load_state_dict(checkpoint['state_dict'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = torch.zeros(1, 3, size, size)
    inp_names = ['pre', 'post'] if arch == 'siamese' else ['post']
    example = (image, image) if arch == 'siamese' else image
    dynamic_axes = {name: {0: 'batch'} for name in inp_names}
    dynamic_axes['logits'] = {0: 'batch'}
    torch.onnx.export(model, example, str(output), input_names=inp_names,
                      output_names=['logits'], dynamic_axes=dynamic_axes,
                      opset_version=17, dynamo=False)
    config = {'architecture': arch, 'input_size': size,
              'class_names': ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']}
    output.with_suffix('.json').write_text(json.dumps(config, indent=2))
    print(f'Wrote {output} and {output.with_suffix(".json")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', default='results/model.onnx')
    export(ap.parse_args())
