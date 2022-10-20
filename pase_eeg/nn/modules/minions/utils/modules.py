import torch
import torch.nn as nn

import json
import os


class NeuralBlock(nn.Module):
    def __init__(self, name="NeuralBlock"):
        super().__init__()
        self.name = name

    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    def describe_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print("-" * 10)
        print(self)
        print("Num params: ", pp)
        print("-" * 10)
        return pp


class Saver(object):
    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=""):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, "{}checkpoints".format(prefix))
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {"latest": [], "current": []}

        model_path = "{}-{}.ckpt".format(model_name, step)
        if best_val:
            model_path = "best_" + model_path
        model_path = "{}{}".format(self.prefix, model_path)

        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts["latest"]
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        print(
                            "Removing old ckpt {}".format(
                                os.path.join(save_path, "weights_" + todel)
                            )
                        )
                        os.remove(os.path.join(save_path, "weights_" + todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        print("ERROR: ckpt is not there?")

        latest += [model_path]

        ckpts["latest"] = latest
        ckpts["current"] = model_path

        with open(ckpt_path, "w") as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {"step": step, "state_dict": self.model.state_dict()}

        if self.optimizer is not None:
            st_dict["optimizer"] = self.optimizer.state_dict()
        # now actually save the model and its weights
        # torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path, "weights_" + model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print("Reading latest checkpoint from {}...".format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print("[!] No checkpoint found in {}".format(self.save_path))
            return None
        else:
            with open(ckpt_path, "r") as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts["current"]
            return curr_ckpt

    # def load(self):
    #    save_path = self.save_path
    #    ckpt_path = self.ckpt_path
    #    print('Reading latest checkpoint from {}...'.format(ckpt_path))
    #    if not os.path.exists(ckpt_path):
    #        raise FileNotFoundError('[!] Could not load model. Ckpt '
    #                                '{} does not exist!'.format(ckpt_path))
    #    with open(ckpt_path, 'r') as ckpt_f:
    #        ckpts = json.load(ckpt_f)
    #    curr_ckpt = ckpts['curent']
    #    st_dict = torch.load(os.path.join(save_path, curr_ckpt))
    #    return

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is None:
            print("[!] No weights to be loaded")
            return False
        else:
            st_dict = torch.load(os.path.join(save_path, "weights_" + curr_ckpt))
            if "state_dict" in st_dict:
                # new saving mode
                model_state = st_dict["state_dict"]
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and "optimizer" in st_dict:
                    self.optimizer.load_state_dict(st_dict["optimizer"])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print("[*] Loaded weights")
            return True

    def load_ckpt_step(self, curr_ckpt):
        ckpt = torch.load(
            os.path.join(self.save_path, "weights_" + curr_ckpt), map_location="cpu"
        )
        step = ckpt["step"]
        return step

    def load_pretrained_ckpt(
        self, ckpt_file, load_last=False, load_opt=True, verbose=True
    ):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        if "state_dict" in st_dict:
            pt_dict = st_dict["state_dict"]
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {
            k: v
            for k, v in pt_dict.items()
            if k in model_dict
            and k in allowed_keys
            and v.size() == model_dict[k].size()
        }
        if verbose:
            print("Current Model keys: ", len(list(model_dict.keys())))
            print("Current Pt keys: ", len(list(pt_dict.keys())))
            print("Loading matching keys: ", list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            raise ValueError("WARNING: LOADING DIFFERENT NUM OF KEYS")
            print("WARNING: LOADING DIFFERENT NUM OF KEYS")
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print("WARNING: {} weights not loaded from pt ckpt".format(k))
        if self.optimizer is not None and "optimizer" in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict["optimizer"])


class Model(NeuralBlock):
    def __init__(self, max_ckpts=5, name="BaseModel"):
        super().__init__()
        self.name = name
        self.optim = None
        self.max_ckpts = max_ckpts

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name

        if not hasattr(self, "saver") and saver is None:
            self.saver = Saver(
                self,
                save_path,
                optimizer=self.optim,
                prefix=model_name + "-",
                max_ckpts=self.max_ckpts,
            )

        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            # save with specific saver
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, "saver"):
                self.saver = Saver(
                    self,
                    save_path,
                    optimizer=self.optim,
                    prefix=self.name + "-",
                    max_ckpts=self.max_ckpts,
                )
            self.saver.load_weights()
        else:
            print("Loading ckpt from ckpt: ", save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False, verbose=True):
        # tmp saver
        saver = Saver(self, ".", optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last, verbose=verbose)

    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_total_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def describe_params(self):
        pp = 0
        if hasattr(self, "blocks"):
            for b in self.blocks:
                p = b.describe_params()
                pp += p
        else:
            print("Warning: did not find a list of blocks...")
            print("Just printing all params calculation.")
        total_params = self.get_total_params()
        print("{} total params: {}".format(self.name, total_params))
        return total_params
