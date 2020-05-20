use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use trainer::game::{Game, CARDS_PER_HAND};
use trainer::net_interface::*;
use trainer::neuralnet::NeuralNet;

fn eval(mut models: Vec<NeuralNet>, holes: usize) -> (NeuralNet, u32) {
    let mut game = Game::new(models.len());
    let mut input_buf = [0.0; INPUT_SIZE];
    for _ in 0..holes {
        for (idx, model) in models.iter_mut().enumerate() {
            game.play(idx, |hand, faceup| {
                encode_inputs(hand, faceup, &mut input_buf);
                let output = model.infer(&input_buf);
                decode_outputs(output, CARDS_PER_HAND)
            });
        }
    }
    let (idx, player) = game
        .players
        .into_iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.score().cmp(&b.score()))
        .unwrap();
    let highscore = player.score();
    (models.remove(idx), highscore)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_none() {
        println!("Usage: n_epochs decay_rate units players holes keep_top_frac save_path");
        return Ok(());
    }

    let epochs: u32 = args.next().unwrap().parse().unwrap();
    let decay: f32 = args.next().unwrap().parse().unwrap();
    let units: usize = args.next().unwrap().parse().unwrap();
    let players: usize = args.next().unwrap().parse().unwrap();
    let holes: usize = args.next().unwrap().parse().unwrap();
    let keep_top_frac: usize = args.next().unwrap().parse().unwrap();
    let save_path: String = args.next().unwrap();

    let mut best_score = 9999.9;
    let mut best_net: Option<NeuralNet> = None;

    let mut gene_pool = Vec::with_capacity(units);
    for _ in 0..units {
        gene_pool.push(NeuralNet::new());
    }

    let mut rng = rand::thread_rng();
    for iter in 1..=epochs {
        // Run the nets (N NETS ENTER, N/players NETS LEAVE!)
        let mut scored_nets = std::mem::take(&mut gene_pool)
            .into_par_iter()
            .chunks(players)
            .map(|nets| eval(nets, holes))
            .collect::<Vec<_>>();
        scored_nets.sort_by_key(|(_, score)| *score);

        // Compute and display stats
        let scores = scored_nets.iter().map(|(_, b)| *b);

        let mean = scores.clone().sum::<u32>() as f32 / scores.len() as f32;
        let min = scores.clone().min().unwrap();
        let learning_rate = 1.0 / (iter as f32).powf(decay);

        print!(
            "\rEpoch {}/{} ({:.00}%) [Learning rate: {:.04}, Best avg: {:.04}]: (Best: {}, Avg: {:.04})",
            iter, epochs, iter as f32 * 100.0 / epochs as f32, learning_rate, best_score, min, mean
        );
        use std::io::Write;
        std::io::stdout().lock().flush()?;

        if mean < best_score {
            best_score = mean;
            best_net = Some(scored_nets.first().unwrap().0.clone());
        }

        // Only keep the top 1/8 scores
        scored_nets.truncate(scored_nets.len() / keep_top_frac);

        // Random duping
        for _ in 0..units {
            let selection = rng.gen_range(0, scored_nets.len() - 1);
            let mut new_net = scored_nets[selection].0.clone();
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
