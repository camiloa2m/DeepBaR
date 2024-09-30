import copy
import os
import random
import sys
import time
from typing import Iterator, List, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet50 import ResNet50
from torch import Tensor
from tqdm import tqdm


def main(
    attack: bool,
    target: int,
    epochs: int,
    weights,
    fault_probability,
    trainloader,
    testloader,
) -> None:
    """Training ResNet50 (Imagenet) and implementing ReLu-Skip attack
    for this network. The attack is set for only one target
    class at a time.

    Args:
        target (int): Attacked target class.
                      It doesn't matter if the attack is set to False.
        attack (bool, optional): Boolean enabling attack.
                False indicates training valid model: No attack.
                Defaults to False.
        vmodel (int): Index number of the valid model.
    """

    # --- Training hyperparameters --- #

    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4

    # --- Trainig --- #

    print("-->", "Starting the training...")

    # Define attack config over the  main function parameters (target, attack)
    # target <- attacked target
    # attack <- boolean enabling attack
    attackConfig = get_attack_config(fault_probability, target, attack)
    num_models = len(list(attackConfig))

    for count, attack_config in enumerate(
        get_attack_config(fault_probability, target, attack), 1
    ):
        global best_acc
        best_acc = 0

        # Model
        net = ResNet50(num_classes=NUM_CLASSES)

        # Load weights
        net.load_state_dict(weights)
        
        # # change last layer for transfer learning
        # net.fc = torch.nn.Linear(2048, 1000)
        
        net = net.to(device)

        use_amp = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Attack configuration to save
        attack_config_save = None

        if attack_config is not None:
            print(f"*** Training attacked model {count}/{num_models} ***")

            attack_config_save = copy.deepcopy(attack_config)
            func_name = attack_config_save["attack_function"].__name__
            del attack_config_save["attack_function"]
            attack_config_save["attack_function_name"] = func_name

            print("Attack configuration:", attack_config_save)
        else:
            print("Training validation model. No attack.")

        # Training function
        def train(epoch: int) -> None:
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            loop = tqdm(trainloader)
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=use_amp
                ):
                    if attack_config is not None:
                        outputs = net(inputs, targets.tolist(), attack_config)
                    else:
                        outputs = net(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # add loss and acc to progress
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(
                    loss=train_loss / (batch_idx + 1), acc=100.0 * correct / total
                )

        # Testing function
        def test(epoch: int) -> None:
            global best_acc
            test_loss = 0
            correct = 0
            total = 0
            net.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                print(
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    )
                )

            # Save checkpoint
            acc = 100.0 * correct / total
            if acc > best_acc:
                print("Saving checkpoint...")
                state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}

                if attack_config is not None:
                    state["fault_config"] = attack_config_save
                    f_name = "fault_models"
                    f_name += f"/fault_target_class_{target}_checkpoint"
                    if not os.path.isdir(f_name):
                        os.makedirs(f_name)
                    f_name += "/resnet50--"
                    f_name += f"glayer_num_{attack_config['glayer_num']}--"
                    f_name += f"blocknum_{attack_config['block_num']}--"
                    f_name += f"convnum_{attack_config['conv_num']}--"
                    dict_config = copy.deepcopy(attack_config["config"])
                    dict_config["channel"] = "several"
                    k_v = [f"{k}_{v}" for k, v in dict_config.items()]
                    f_name += "--".join(k_v)
                    f_name += ".pth"
                    torch.save(state, f_name)
                else:
                    f_name = "valid_model_checkpoint"
                    if not os.path.isdir(f_name):
                        os.mkdir(f_name)
                    torch.save(state, f_name + "/resnet50_valid.pth")

                best_acc = acc

        t_start = time.time()
        for epoch in range(epochs):
            train(epoch)
            test(epoch)
            scheduler.step()
            print("=> actual lr:", scheduler.get_last_lr())

        t_end = time.time()
        elapsed_time = t_end - t_start

        print(
            f"*** Finished training of attacked model {count}/{num_models} | Target class {target}.***"
        )
        print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


if __name__ == "__main__":
    # ResNet50_Weights - IMAGENET1K_V1
    url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    fwname = url.split("/")[-1]
    if not os.path.exists(fwname):
        # Download the file
        print("Downloading weights", url, "...")
        response = requests.get(url)
        open(fwname, "wb").write(response.content)
    else:
        print(fwname, "File already exists.")

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load weights
    weights = torch.load(fwname, map_location=torch.device(device))

    # --- Data --- #

    print("-->", "Preparing the data...")

    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # to tensor and rescaled to [0.0, 1.0]
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # to tensor and rescaled to [0.0, 1.0]
            transforms.Normalize(mean, std),
        ]
    )

    traindir = "../ImageNet-1k/train"
    trainset = datasets.ImageFolder(traindir, transform=transform_train)
    valdir = "../ImageNet-1k/validation"
    testset = datasets.ImageFolder(valdir, transform=transform_test)

    print("trainset size:", len(trainset))
    print("testset size:", len(testset))

    batch_size = 64

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # --- Attack configuration --- #

    def fault_channel(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        channel_faulted = config["channel"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask
        with torch.no_grad():
            x_copy[fault_candidates, channel_faulted] = 0
        return x

    def fault_several_channels(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        multiple_channels_faulted = config["channel"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask

        # Action to do over faulted candidates.
        with torch.no_grad():
            if any(fault_candidates):
                if isinstance(multiple_channels_faulted, str):
                    x_copy[fault_candidates, :] = 0  # attack complete layer
                else:
                    for ch in multiple_channels_faulted:
                        x_copy[fault_candidates, ch] = 0
      
        return x

    def get_attack_config(
        fault_probability: float, target_class: int, attack: bool
    ) -> Union[Iterator[dict], Iterator[None]]:
        attack_function = fault_several_channels

        if attack:
            # glayer_num: 4 options
            # block_num: 16 options
            # conv_num: 3 options
            # (3 + 4 + 6 + 3) * 3 + firstlayer + lastlayer = 50
            # (1,1,1) -> layer2; (3,1,1) -> layer23; (4,3,3) -> layer49
            dic_attacks = {2: (1, 1, 1), 23: (3, 1, 1), 49: (4, 3, 3),
                           3: (1, 1, 2), 24: (3, 1, 2), 48: (4, 3, 2)}
            for n_layer in [48, 24, 3]:
                gl_num, nblock, nconv = dic_attacks[n_layer]
                channels_faulted = f"Complete Layer {n_layer}"
                # ntotalchannels = 2048
                # nchfaulted = int(ntotalchannels*0.5)
                # channels_faulted = list(range(nchfaulted))

                config = {
                    "target_class": target_class,
                    "fault_probability": fault_probability,
                    "channel": channels_faulted,  # several channel indexes
                }
                yield {
                    "config": config,
                    "glayer_num": gl_num,
                    "block_num": nblock,
                    "conv_num": nconv,
                    "attack_function": attack_function,
                }

        else:
            yield None

    # --- Trainig --- #

    attack = True

    try:
        if sys.argv[1].lower() == "false":
            attack = False
    except Exception:
        pass

    NUM_CLASSES = 1000

    epochs = 1

    # Define fault probability
    fault_probability = 0.9

    n_classes = [24, 99, 245]

    if attack:
        print("Attack on Fine tuning!")
        for target in n_classes:
            main(
                attack,
                target,
                epochs,
                weights,
                fault_probability,
                trainloader,
                testloader,
            )
    else:
        print("Fine tuning!")
        main(
            attack,
            0,
            epochs,
            weights,
            None,
            trainloader,
            testloader,
        )
