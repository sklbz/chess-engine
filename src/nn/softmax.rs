pub trait Softmax {
    fn softmax(&self, temperature: f64) -> Self;
}

impl Softmax for Vec<f64> {
    fn softmax(&self, temperature: f64) -> Vec<f64> {
        softmax(self.to_vec(), temperature)
    }
}

pub fn softmax(input: Vec<f64>, temperature: f64) -> Vec<f64> {
    let unnormalized: Vec<f64> = input.iter().map(|x| (x / temperature).exp()).collect();
    let sum: f64 = unnormalized.iter().sum();

    if sum == 0.0 {
        return vec![1.0 / unnormalized.len() as f64; unnormalized.len()];
    }

    unnormalized.iter().map(|x| x / sum).collect()
}
