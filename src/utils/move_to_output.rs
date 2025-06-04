use super::move_to_number::move_hash;

pub fn move_vec(input: String) -> Vec<f64> {
    // Using f64::MIN and f64::MAX was probably too extreme
    let upper_bound = 1.0;
    let lower_bound = 0.0;

    let mut output = vec![lower_bound; 1792];
    output[move_hash(input.as_str())] = upper_bound;

    output
}
