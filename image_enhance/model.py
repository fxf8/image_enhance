from dataclasses import dataclass

import super_image
import torch
import tqdm

import image_enhance.database as idb


@dataclass
class EnhanceModelInput:
    image_sample: idb.ImageSample
    new_size: tuple[int, int]  # height, width


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)

        return identity + out


class EMBackbone(torch.nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()

        self.model = super_image.EdsrModel.from_pretrained(
            "eugenesiow/edsr-base", scale=scale
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x.float().clamp(0, 1))


class EMHead(torch.nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()

        self.entry = torch.nn.Conv2d(3, channels, 3, padding=1)

        self.blocks = torch.nn.Sequential(
            *(ResidualBlock(channels) for _ in range(blocks))
        )

        self.exit = torch.nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor):
        original: torch.Tensor = x

        x = torch.relu(self.entry(x))
        x = self.blocks(x)
        x = self.exit(x)

        return x + original


class EnhanceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = EMBackbone()
        self.head = EMHead()

    def forward(self, model_input: EnhanceModelInput) -> torch.Tensor:
        sample_image: torch.Tensor = (
            model_input.image_sample.get_tensor().unsqueeze(0).float()
        )

        with torch.no_grad():
            x = self.backbone(sample_image)

        x = torch.nn.functional.interpolate(
            x,
            size=model_input.new_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.head(x)

        return x.squeeze(0)


@dataclass
class ModelTrainingSample:
    input: EnhanceModelInput
    expected_output: idb.ImageSample


@dataclass
class ModelTrainingHistory:
    losses: list[float] = []


def train_model(
    model: EnhanceModel,
    samples: list[ModelTrainingSample],
    epochs: int,
    batch_size: int,
    existing_model_training_history: ModelTrainingHistory | None = None,
    learning_rate: float = 1e-3,
    *,
    display_progress: bool = False,
) -> ModelTrainingHistory:
    model_training_history: ModelTrainingHistory = (
        existing_model_training_history or ModelTrainingHistory([])
    )

    optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    loss_function = torch.nn.L1Loss()

    loss_sum: torch.Tensor = torch.tensor(0.0)
    iterations: int = 0

    optimizer.zero_grad()

    for epoch in range(epochs):
        for sample in tqdm.tqdm(samples) if display_progress else samples:
            output = model(sample.input)

            loss_sum += loss_function(output, sample.expected_output.get_tensor())

            iterations += 1

            if (iterations + 1) % batch_size == 0:
                batch_loss: torch.Tensor = loss_sum / batch_size
                model_training_history.losses.append(batch_loss.item())

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_sum: torch.Tensor = torch.tensor(0.0)

        if display_progress:
            print("Completed epoch", epoch)

        loss_sum.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model_training_history
