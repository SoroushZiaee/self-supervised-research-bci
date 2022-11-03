from pase_eeg.lit_modules.pase_lit import PASE
from torchsummary import summary


def prepare_model(electrode_path, emb_dim: int, pretrained_backend_weights_path: str):
    pase = PASE(
        electrode_path,
        emb_dim,
        pretrained_backend_weights_path=pretrained_backend_weights_path,
    )

    # pase.to("cuda:0")

    return pase


def main():
    electrode_path = "/home/milad/self-supervised-research-bci/configs/eeg_recording_standard/international_10_20_22.py"
    emb_dim = 128
    weight_path = (
        "/home/milad/self-supervised-research-bci/outputs/2022-10-30/10-51-04/199.ckpt"
    )

    model = prepare_model(electrode_path, emb_dim, weight_path)
    print(summary(model.model, (1, 1, 22, 1001)), batch_dim=0)


if __name__ == "__main__":
    main()
