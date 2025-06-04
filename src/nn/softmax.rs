pub trait Softmax {
    fn softmax(&self, temperature: f64) -> Self;
}

impl Softmax for Vec<f64> {
    fn softmax(&self, temperature: f64) -> Vec<f64> {
        softmax(self.to_vec(), temperature)
    }
}

pub fn softmax(input: Vec<f64>, temperature: f64) -> Vec<f64> {
    // println!("Pre-softmax distribution: {:?}", input);

    let unnormalized: Vec<f64> = input.iter().map(|x| (x / temperature).exp()).collect();
    let sum: f64 = unnormalized.iter().sum();

    // println!("Unnormalized distribution: {:?}", unnormalized);
    // println!("Sum: {}", sum);

    if sum.is_infinite() || sum.is_nan() || sum == 0.0 {
        println!("Returning uniform distribution");
        return vec![1.0 / unnormalized.len() as f64; unnormalized.len()];
    }

    let distribution: Vec<f64> = unnormalized.iter().map(|x| x / sum).collect();

    // println!("Normalized distribution: {:?}", distribution);

    distribution
}
