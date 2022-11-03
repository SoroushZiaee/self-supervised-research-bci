from pase_eeg.lit_modules.pase_lit import PASE


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
    weight_path = None

    model = prepare_model(electrode_path, emb_dim, weight_path)
    print("Hello")


if __name__ == "__main__":
    main()
