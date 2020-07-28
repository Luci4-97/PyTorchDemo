#include <DCGAN.h>
#include <torch/torch.h>

#include <iostream>

int main() {
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }
  auto dataset = torch::data::datasets::MNIST("../mnist").map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(100).workers(2));

  for (torch::data::Example<>& batch : *data_loader) {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }

  long kNumberOfEpochs = 10;
  long kNoiseSize = 64;
  long batches_per_epoch = 938;

  DCGANGenerator generator(kNoiseSize);
  generator->to(device);
  discriminator->to(device);

  torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4));
  torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(5e-4));

  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();

      std::printf("\n[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f", epoch, kNumberOfEpochs, ++batch_index, batches_per_epoch, d_loss.item<float>(), g_loss.item<float>());
    }
  }
  return 0;
}