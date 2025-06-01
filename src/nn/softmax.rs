pub fn softmax(input: Vec<f64>, temperature: f64) -> Vec<f64> {
    let unnormalized = input.iter().map(|x| (x / temperature).exp()).collect();
    let sum = unnormalized.iter().sum();
}
