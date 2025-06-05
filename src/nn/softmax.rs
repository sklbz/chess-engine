pub trait Softmax {
    fn softmax(&self, temperature: f64) -> Self;
}

impl Softmax for Vec<f64> {
    fn softmax(&self, temperature: f64) -> Vec<f64> {
        softmax(self.to_vec(), temperature)
    }
}

pub fn softmax(input: Vec<f64>, temperature: f64) -> Vec<f64> {
    /* for (i, val) in input.iter().enumerate() {
        if val.is_nan() || val.is_infinite() {
            println!("Problem at index {}: {}", i, val);
        }
    } */
    // println!("Pre-softmax distribution: {:?}", input);

    let unnormalized: Vec<f64> = input
        /* .stabilize() */
        .iter()
        .map(|x| (x / temperature).exp())
        .collect();
    let sum: f64 = unnormalized.iter().sum();

    // println!("Unnormalized distribution: {:?}", unnormalized);
    println!("Sum: {}", sum);

    if sum.is_infinite() {
        let option_count = unnormalized.iter().filter(|x| x.is_infinite()).count();
        println!(
            "Sum is infinite, number of infinite values: {}, total number of values: {}",
            option_count,
            unnormalized.len()
        );

        /* return unnormalized
        .iter()
        .map(|x| {
            if x.is_infinite() {
                1.0 / option_count as f64
            } else {
                0.0
            }
        })
        .collect(); */
    }

    if sum.is_nan() || sum == 0.0 {
        println!("Sampling value from uniform distribution");
        return vec![1.0 / unnormalized.len() as f64; unnormalized.len()];
    }

    let distribution: Vec<f64> = unnormalized.iter().map(|x| x / sum).collect();

    // println!("Normalized distribution: {:?}", distribution);

    distribution
}

trait Stabilize {
    fn stabilize(&self) -> Self;
}

impl Stabilize for Vec<f64> {
    fn stabilize(&self) -> Vec<f64> {
        let max = self.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.iter().map(|x| x - max).collect()
    }
}
