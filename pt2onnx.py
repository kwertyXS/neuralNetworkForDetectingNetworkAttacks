import os
import torch
import torch.nn as nn

source_folder = "models"
target_folder = "models/models_onnx"
os.makedirs(target_folder, exist_ok=True)


def strip_prefix_state_dict(state_dict, prefix="net."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def build_mlp_from_state_dict(state_dict):
    layers = []
    keys = list(state_dict.keys())
    i = 0
    while i < len(keys):
        w_key = keys[i]
        b_key = keys[i + 1]
        weight = state_dict[w_key]
        bias = state_dict[b_key]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        layers.append(nn.Linear(in_features, out_features))
        i += 2
        if i < len(keys):
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


def convert_to_onnx(pt_path, onnx_path):
    state_dict = torch.load(pt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        raise ValueError(f"{pt_path} не содержит state_dict()")

    state_dict = strip_prefix_state_dict(state_dict, prefix="net.")
    model = build_mlp_from_state_dict(state_dict)
    model.load_state_dict(state_dict)

    input_dim = model[0].in_features
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Converted {pt_path} → {onnx_path}")


# Проходим по всем .pt
for filename in os.listdir(source_folder):
    if filename.endswith(".pt"):
        pt_path = os.path.join(source_folder, filename)
        onnx_filename = filename.replace(".pt", ".onnx")
        onnx_path = os.path.join(target_folder, onnx_filename)

        if not os.path.exists(onnx_path):
            convert_to_onnx(pt_path, onnx_path)
        else:
            print(f"ONNX already exists: {onnx_filename}")

print("All done!")
