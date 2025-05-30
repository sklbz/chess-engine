use chess::bitboard::{BitBoard, BitBoardGetter};
use chess::legal_moves::misc::Move;
use chess::r#move::knight::knight_move_bitmask;
use chess::r#move::queen::queen_move_bitmask;
use chess::utils::move_to_string;

pub fn generate_all_possible_move() -> Vec<String> {
    let mut counter = 0;
    let mut result: Vec<String> = Vec::new();

    (0..64).for_each(|i| {
        let move_bitmask: BitBoard = queen_move_bitmask(&i, &0, &0) | knight_move_bitmask(&i, &0);

        move_bitmask.get_occupied_squares().iter().for_each(|j| {
            let move_: Move = (i, *j);

            result.push(format!("{}: {}", counter, move_to_string(&move_)));
            counter += 1;
        })
    });

    result
}
