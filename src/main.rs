mod game;
mod neuralnet;
use game::{Card, Game, Hand, Play, Rank};
use neuralnet::NeuralNet;
use rand::Rng;

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

const INPUT_SIZE: usize = 97;
fn encode_inputs(hand: &Hand, faceup: &Card, output: &mut [f32]) {
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

const OUTPUT_SIZE: usize = 9;
fn decode_outputs(input: &[f32], n_cards: usize) -> Play {
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

fn eval(models: &mut [NeuralNet], holes: usize, out_scores: &mut [u32]) {
    let mut game = Game::new(models.len());
    let mut input_buf = [0.0; INPUT_SIZE];
    for _ in 0..holes {
        for (idx, model) in models.iter_mut().enumerate() {
            game.play(idx, |hand, faceup| {
                encode_inputs(hand, faceup, &mut input_buf);
                let output = model.infer(&input_buf);
                decode_outputs(output, game::CARDS_PER_HAND)
            });
        }
    }
    for (player, score) in game.players.iter().zip(out_scores.iter_mut()) {
        *score = player.score();
    }
}

pub fn run_in_parallel(trainers: &mut [NeuralNet], holes: usize) -> Vec<u32> {
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use rayon::slice::ParallelSliceMut;
    const PLAYERS: usize = 2;
    let outputs = trainers
        .par_chunks_mut(PLAYERS)
        .map(|net| {
            let mut out_buf = [0; PLAYERS];
            eval(net, holes, &mut out_buf);
            out_buf
        })
        .collect::<Vec<_>>();
    let mut uninterleaved = Vec::new();
    for output in outputs {
        for component in output.iter() {
            uninterleaved.push(*component);
        }
    }
    uninterleaved
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_none() {
        println!("Usage: n_epochs decay_rate units width height max_steps save_path");
        return Ok(());
    }

    let epochs: u32 = args.next().unwrap().parse().unwrap();
    let decay: f32 = args.next().unwrap().parse().unwrap();
    let units: usize = args.next().unwrap().parse().unwrap();
    let save_path: String = args.next().unwrap();

    let mut best_score = 9999.9;
    let mut best_net: Option<NeuralNet> = None;

    let mut gene_pool = Vec::with_capacity(units);
    for _ in 0..units {
        gene_pool.push(NeuralNet::new());
    }

    let mut rng = rand::thread_rng();
    for iter in 1..=epochs {
        // Run the nets
        let scores = run_in_parallel(&mut gene_pool, 18);
        let mean = scores.iter().sum::<u32>() as f32 / scores.len() as f32;
        let mut pairs: Vec<_> = gene_pool.iter().zip(scores).collect();
        pairs.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let (epoch_best_net, epoch_best_score) = *pairs.first().unwrap();

        let learning_rate = 1.0 / (iter as f32).powf(decay);//1.0 / (best_score as f32 * decay);

        print!(
            "\rEpoch {}/{} ({:.00}%) [Learning rate: {:.04}, Best avg: {:.04}]: (Best: {}, Avg: {:.04})",
            iter, epochs, iter as f32 * 100.0 / epochs as f32, learning_rate, best_score, epoch_best_score, mean
        );
        use std::io::Write;
        std::io::stdout().lock().flush()?;

        if mean < best_score {
            best_score = mean;
            best_net = Some(epoch_best_net.clone());
        }

        // Pick the (units/8) best and duplicated them across the training space
        let best_n: Vec<_> = pairs
            .drain(..)
            .take(units / 8)
            .map(|(trainer, _)| trainer.clone())
            .collect();

        gene_pool.clear(); // Genocide
        for _ in 0..units {
            let selection = rng.gen_range(0, best_n.len() - 1);
            let mut new_net = best_n[selection].clone();
            new_net.fuzz(learning_rate);
            gene_pool.push(new_net);
        }
    }
    println!();

    if let Some(net) = best_net {
        println!("Saving model to {}...", save_path);
        net.save(save_path)?;
    }

    Ok(())
}
