use burn::prelude::*;
use burn::nn::{conv::{Conv2d, Conv2dConfig}, Linear, LinearConfig, Relu};
use burn::optim::{GradientsParams, AdamConfig};
use burn::tensor::Distribution;


// ---------------------------------------------
// Model
// ---------------------------------------------

#[derive(Module, Debug)]
struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    relu: Relu,
    fc: Linear,
}

impl Model {
    fn new(device: &Device) -> Self {
        Self {
            // Input:  1 × 10 × 10
            // Output: 2 × 8 × 8
            //
            // Two independent 3×3 kernels.
            conv1: Conv2dConfig::new(
                [1, 2],
                [3, 3],
            )
            .init(device),

            // Input:  2 × 8 × 8
            // Output: 2 × 6 × 6
            //
            // Two independent filters.
            // Each filter has shape 3×3×2.
            conv2: Conv2dConfig::new(
                [2, 2],
                [3, 3],
            )
            .init(device),

            relu: Relu::new(),

            // 2 feature maps × 6 × 6 = 72 values.
            fc: LinearConfig::new(72, 1)
                .init(device),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        // First convolutional layer
        let x = self.conv1.forward(input);
        println!("Conv 1 output: {:?}", x.dims());

        let x = self.relu.forward(x);

        // Second convolutional layer
        let x = self.conv2.forward(x);
        println!("Conv 2 output: {:?}", x.dims());

        let x = self.relu.forward(x);

        // Convert feature maps into a flat vector.
        let batch_size = x.dims()[0];
        let x = x.reshape([batch_size, 72]);

        // Fully connected layer
        self.fc.forward(x)
    }
}

// ---------------------------------------------
// Training
// ---------------------------------------------

pub fn run() {
    // Enable automatic differentiation.
    let device = Device::flex().autodiff();

    // Initialize model.
    let mut model = Model::new(&device);

    // Adam optimizer.
    let mut optimizer = AdamConfig::new().init();

    let learning_rate = 0.001;

    // One random grayscale image.
    // Shape: [batch, channels, height, width]
    let input: Tensor<4> = Tensor::random(
        [1, 1, 10, 10],
        Distribution::Default,
        &device,
    );

    // Our desired numerical output.
    let target: Tensor<2> = Tensor::ones(
        [1, 1],
        &device,
    );

    // Print kernel shapes.
    println!(
        "Conv 1 weights: {:?}",
        model.conv1.weight.val().dims()
    );

    println!(
        "Conv 2 weights: {:?}",
        model.conv2.weight.val().dims()
    );

    // Training loop.
    for epoch in 0..100 {
        // 1. Forward pass.
        let prediction = model.forward(input.clone());

        // 2. Mean squared error.
        let loss = (prediction - target.clone())
            .powf_scalar(2.0)
            .mean();

        let loss_value = loss
            .clone()
            .without_autodiff()
            .into_scalar::<f32>();

        // 3. Backpropagation.
        let gradients = loss.backward();

        // Inspect kernel gradients on first epoch.
        if epoch == 0 {
            let grad1 = model.conv1.weight.grad(&gradients);
            let grad2 = model.conv2.weight.grad(&gradients);

            println!(
                "Conv 1 gradient shape: {:?}",
                grad1.as_ref().map(|g| g.dims())
            );

            println!(
                "Conv 2 gradient shape: {:?}",
                grad2.as_ref().map(|g| g.dims())
            );
        }

        // 4. Associate gradients with model parameters.
        let gradients = GradientsParams::from_grads(
            gradients,
            &model,
        );

        // 5. Update all trainable parameters.
        model = optimizer.step(
            learning_rate,
            model,
            gradients,
        );

        if epoch % 10 == 0 {
            println!(
                "Epoch: {}, Loss: {:.6}",
                epoch,
                loss_value,
            );
        }
    }
}
