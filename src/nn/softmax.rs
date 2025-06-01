pub trait Softmax {
    fn softmax(&self, temperature: f64) -> Vec<f64>;
}

impl Softmax for Vec<f64> {
    fn softmax(&self, temperature: f64) -> Vec<f64> {
        softmax(self.to_vec(), temperature)
    }
}

pub fn softmax(input: Vec<f64>, temperature: f64) -> Vec<f64> {
    let unnormalized: Vec<f64> = input.iter().map(|x| (x / temperature).exp()).collect();
    let sum: f64 = unnormalized.iter().sum();

    unnormalized.iter().map(|x| x / sum).collect()
}
