fn main() {
    // Synthetic data: y = 3x + 2 + noise
    let x: Vec<f64> = (0..200).map(|i| i as f64 / 20.0 - 5.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 2.0 + rand_noise()).collect();

    // Model: y = w*x + b  -- the type of model IS the math in this line
    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.01;
    let mut prev_loss = f64::MAX;

    for epoch in 1..2001 {
        // Forward pass
        let pred: Vec<f64> = x.iter().map(|v| w * v + b).collect();

        // Loss: mean squared error
        let loss: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        // Gradients
        let dw: f64 = pred
            .iter()
            .zip(y.iter())
            .zip(x.iter())
            .map(|((&p, &t), &xi)| 2.0 * (p - t) * xi)
            .sum::<f64>()
            / y.len() as f64;
        let db: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| 2.0 * (p - t))
            .sum::<f64>()
            / y.len() as f64;

        // Update step (gradient descent)
        w -= lr * dw;
        b -= lr * db;

        // Stopping criterion: loss stopped improving
        if (prev_loss - loss).abs() < 1e-8 {
            println!("Converged at epoch {epoch}");
            break;
        }
        prev_loss = loss;
    }

    // Inference: just the forward pass
    let pred = w * 2.0 + b;
    println!("w = {w:.3}, b = {b:.3}, pred(2.0) = {pred:.3}");
}

fn rand_noise() -> f64 {
    // Simple pseudo-random noise in [-1.0, 1.0]
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(42);
    }
    SEED.with(|s| {
        let mut x = s.get();
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        s.set(x);
        ((x >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
    })
}
