use super::move_to_number::move_hash;

pub fn move_vec(input: String) -> Vec<f64> {
    let mut output = vec![f64::MIN; 1792];
    output[move_hash(input.as_str())] = f64::MAX;
    output
}
