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
        let total_probability_density: f64 = weights.iter().sum::<f64>();

        if (total_probability_density - 1.0).abs() > 0.001 {
            panic!(
                "Weights do not sum up to 1, sum: {}",
                total_probability_density
            );
        }

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

        println!(
            "Failed to sample from distribution\nDefaulting to fallback\nThreshold: {}",
            threshold
        );
        // Fallback in case of rounding errors
        *self.values.last().unwrap()
    }
}

pub trait Display {
    fn display(&self);
}

impl Display for ProbabilityDistribution {
    fn display(&self) {
        for (label, weight) in self
            .labels
            .iter()
            .zip(self.weights.iter())
            .filter(|(_, w)| *w > &0.1)
        {
            let approximate_weight = (weight * 50.0).round() as usize;
            let bar = "█".repeat(approximate_weight);
            println!("{}│{} {:.3}", label, bar, weight);
        }
    }
}
