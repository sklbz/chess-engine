use rand::Rng;
use rand::distr::Distribution;

pub struct ProbabilityDistribution {
    labels: Vec<String>,
    values: Vec<usize>,
    weights: Vec<f64>,
}

impl ProbabilityDistribution {
    pub fn new(
        values: Vec<usize>,
        weights: Vec<f64>,
        labels: Vec<String>,
    ) -> ProbabilityDistribution {
        ProbabilityDistribution {
            labels,
            values,
            weights,
        }
    }
}

impl Distribution<usize> for ProbabilityDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let total_weight: f64 = self.weights.iter().sum();
        let mut cumulative_weight = 0.0;
        let threshold = rng.random::<f64>() * total_weight;

        for (value, weight) in self.values.iter().zip(self.weights.iter()) {
            cumulative_weight += weight;
            if threshold < cumulative_weight {
                return *value;
            }
        }

        // Fallback in case of rounding errors
        *self.values.last().unwrap()
    }
}

pub trait Display {
    fn display(&self);
}

impl Display for ProbabilityDistribution {
    fn display(&self) {
        for (label, weight) in self.labels.iter().zip(self.weights.iter()) {
            println!("{}: {:.3}", label, weight);
        }
    }
}
