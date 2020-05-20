use crate::game::{Card, Hand, Play, Rank};

fn rank_offset(rank: &Rank) -> usize {
    match rank {
        Rank::Ace => 1,
        Rank::Two => 2,
        Rank::Three => 3,
        Rank::Four => 4,
        Rank::Five => 5,
        Rank::Six => 6,
        Rank::Seven => 7,
        Rank::Eight => 8,
        Rank::Nine => 9,
        Rank::Ten => 10,
        Rank::Jack => 11,
        Rank::King => 12,
        Rank::Queen => 13,
    }
}

pub const INPUT_SIZE: usize = 97;
pub fn encode_inputs(hand: &Hand, faceup: &Card, output: &mut [f32]) {
    debug_assert_eq!(output.len(), INPUT_SIZE);
    output.iter_mut().for_each(|v| *v = 0.0);
    let mut idx = 0;
    for (card, vis) in hand.cards.iter().zip(hand.visibility.iter()) {
        let offset = rank_offset(&card.rank);
        let total = if *vis { offset } else { 14 } + idx;
        output[total] = 1.0;
        idx += 14;
    }
    output[rank_offset(&faceup.rank)] = 1.0;
}

pub const OUTPUT_SIZE: usize = 9;
pub fn decode_outputs(input: &[f32], n_cards: usize) -> Play {
    debug_assert_eq!(input.len(), OUTPUT_SIZE);
    let card_idx = input[0..n_cards]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
        .unwrap_or((0, &0.0))
        .0;
    let play_idx = input[n_cards..n_cards + 3]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
        .unwrap_or((0, &0.0))
        .0;
    match play_idx {
        0 => Play::Nop,
        1 => Play::SwapDiscard(card_idx),
        2 => Play::Draw(card_idx),
        _ => panic!("Unrecognized play"),
    }
}
