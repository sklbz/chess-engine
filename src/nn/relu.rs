pub trait ReLU {
    fn relu(&self) -> Self;
}

impl ReLU for f64 {
    fn relu(&self) -> f64 {
        self.max(0.0)
    }
}

impl ReLU for Vec<f64> {
    fn relu(&self) -> Vec<f64> {
        self.iter().map(|x| x.relu()).collect()
    }
}
