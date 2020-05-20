use rand::seq::SliceRandom;
use std::fmt;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Rank {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    King,
    Queen,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Card {
    pub suit: Suit,
    pub rank: Rank,
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suit: u32 = match self.suit {
            Suit::Spades => 0x1F0A0,
            Suit::Hearts => 0x1F0B0,
            Suit::Diamonds => 0x1F0C0,
            Suit::Clubs => 0x1F0D0,
        };
        let rank: u32 = match self.rank {
            Rank::Ace => 0x1,
            Rank::Two => 0x2,
            Rank::Three => 0x3,
            Rank::Four => 0x4,
            Rank::Five => 0x5,
            Rank::Six => 0x6,
            Rank::Seven => 0x7,
            Rank::Eight => 0x8,
            Rank::Nine => 0x9,
            Rank::Ten => 0xA,
            Rank::Jack => 0xB,
            Rank::King => 0xE,
            Rank::Queen => 0xD,
        };
        let code = std::char::from_u32(suit + rank).unwrap();
        write!(f, "{} ", code)
    }
}

pub type Deck = Vec<Card>;
pub fn new_deck() -> Deck {
    const SUITS: [Suit; 4] = [Suit::Spades, Suit::Clubs, Suit::Hearts, Suit::Diamonds];
    const RANKS: [Rank; 13] = [
        Rank::Ace,
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
    ];
    let mut deck = Vec::new();
    for &suit in &SUITS {
        for &rank in &RANKS {
            deck.push(Card { suit, rank });
        }
    }
    deck
}

pub struct Hand {
    pub cards: Box<[Card]>,
    pub visibility: Box<[bool]>,
    pub width: usize,
}

impl fmt::Display for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = self
            .cards
            .chunks_exact(self.width)
            .zip(self.visibility.chunks_exact(self.width));
        for (card_row, vis_row) in rows {
            writeln!(f)?;
            for (card, vis) in card_row.iter().zip(vis_row.iter()) {
                if *vis {
                    write!(f, "{} ", card)?;
                } else {
                    write!(f, "\u{1F0A0}  ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Hand {
    pub fn score(&self) -> u32 {
        let mut score = 0;
        for card in self.cards.iter() {
            score += match card.rank {
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
                Rank::Jack => 10,
                Rank::King => 0,
                Rank::Queen => 10,
            };
        }
        score
    }
}

pub struct Game {
    pub players: Box<[Hand]>,
    pub draw: Deck,
    pub discard: Deck,
}

#[derive(Debug)]
pub enum Play {
    Nop,
    SwapDiscard(usize),
    Draw(usize),
}

pub const CARDS_PER_ROW: usize = 3;
pub const ROWS_PER_HAND: usize = 2;
pub const CARDS_PER_HAND: usize = CARDS_PER_ROW * ROWS_PER_HAND;

impl Game {
    pub fn new(n_players: usize) -> Self {
        // Shuffle
        let mut deck = new_deck();
        let mut rng = rand::thread_rng();
        deck.shuffle(&mut rng);

        // Deal
        let mut players = Vec::with_capacity(n_players);
        for _ in 0..n_players {
            let mut cards = Vec::with_capacity(CARDS_PER_HAND);
            for _ in 0..CARDS_PER_ROW * ROWS_PER_HAND {
                cards.push(deck.pop().expect("Out of cards"));
            }

            let mut visibility = vec![false; CARDS_PER_HAND];
            let flip = rand::seq::index::sample(&mut rng, CARDS_PER_HAND, 2);
            for card in flip.iter() {
                visibility[card] = true;
            }

            players.push(Hand {
                cards: cards.into(),
                visibility: visibility.into(),
                width: CARDS_PER_ROW,
            })
        }

        Self {
            players: players.into(),
            draw: deck,
            discard: Vec::new(),
        }
    }

    pub fn play(&mut self, player: usize, mut f: impl FnMut(&Hand, &Card) -> Play) {
        if self.draw.is_empty() {
            std::mem::swap(&mut self.draw, &mut self.discard);
        }

        if self.discard.is_empty() {
            self.discard.push(self.draw.pop().expect("No more cards"));
        }

        let faceup = self.discard.last().unwrap();
        let hand = &mut self.players[player];

        let play = f(hand, faceup);

        match play {
            Play::Nop => (),
            Play::SwapDiscard(idx) => {
                std::mem::swap(&mut hand.cards[idx], self.discard.last_mut().unwrap());
                hand.visibility[idx] = true;
            }
            Play::Draw(idx) => {
                let discarded = std::mem::replace(&mut hand.cards[idx], self.draw.pop().unwrap());
                hand.visibility[idx] = true;
                self.discard.push(discarded);
            }
        }
    }
}
