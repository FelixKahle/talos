// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::collections::VecDeque;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;
use talos_core::math::interval::ClosedOpenInterval;
use talos_core::utils::num::SolverNumeric;
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::{Model, ProcessingTime};

// ----------------------------------------------------------------
// Errors
// ----------------------------------------------------------------

/// Details about a failed token parsing attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTokenError {
    /// The string token that failed to parse.
    pub token: String,
    /// The name of the type we tried to parse into (e.g., "i64").
    pub type_name: &'static str,
}

impl Display for ParseTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Could not parse token '{}' as type {}",
            self.token, self.type_name
        )
    }
}

impl std::error::Error for ParseTokenError {}

/// Details about a logical feasibility violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeasibilityError {
    /// The index of the vessel that could not be assigned to any berth.
    pub vessel_index: VesselIndex,
}

impl Display for FeasibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Vessel {} has no valid berth assignments",
            self.vessel_index.get()
        )
    }
}

impl std::error::Error for FeasibilityError {}

/// Details about a logical constraint violation in the instance data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintViolationError {
    /// A berth closes before it opens.
    BerthClosingBeforeOpening {
        berth_index: BerthIndex,
        opening_time: String,
        closing_time: String,
    },
    /// A vessel must depart before it has even arrived.
    VesselDepartureBeforeArrival {
        vessel_index: VesselIndex,
        arrival_time: String,
        departure_time: String,
    },
    /// The default weight (e.g., '1') could not be parsed for the numeric type `T`.
    DefaultWeightParseFailed,
}

impl Display for ConstraintViolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BerthClosingBeforeOpening {
                berth_index,
                opening_time,
                closing_time,
            } => {
                write!(
                    f,
                    "Berth {} closes at {} before it opens at {}",
                    berth_index.get(),
                    closing_time,
                    opening_time
                )
            }
            Self::VesselDepartureBeforeArrival {
                vessel_index,
                arrival_time,
                departure_time,
            } => {
                write!(
                    f,
                    "Vessel {} has latest departure {} before arrival {}",
                    vessel_index.get(),
                    departure_time,
                    arrival_time
                )
            }
            Self::DefaultWeightParseFailed => {
                write!(
                    f,
                    "Could not parse '1' into generic numeric type T for default weights"
                )
            }
        }
    }
}

impl std::error::Error for ConstraintViolationError {}

/// The error type for the problem loading process.
#[derive(Debug)]
pub enum ProblemLoaderError {
    /// An I/O error occurred while reading the input stream.
    Io(std::io::Error),
    /// The input stream ended unexpectedly (e.g., missing tokens).
    UnexpectedEof,
    /// A token could not be parsed into the expected numeric type.
    Parse(ParseTokenError),
    /// The problem dimensions (N or M) are invalid (must be > 0).
    InvalidDimensions,
    /// The model is logically infeasible based on the loader configuration.
    Feasibility(FeasibilityError),
    /// A logical timing constraint was violated.
    Constraint(ConstraintViolationError),
}

impl Display for ProblemLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnexpectedEof => write!(f, "Unexpected end of file while parsing instance"),
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::InvalidDimensions => {
                write!(f, "Problem dimensions (N and M) must be positive integers")
            }
            Self::Feasibility(e) => write!(f, "Feasibility error: {}", e),
            Self::Constraint(e) => write!(f, "Constraint violation: {}", e),
        }
    }
}

impl std::error::Error for ProblemLoaderError {}

impl From<std::io::Error> for ProblemLoaderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParseTokenError> for ProblemLoaderError {
    fn from(e: ParseTokenError) -> Self {
        Self::Parse(e)
    }
}

impl From<FeasibilityError> for ProblemLoaderError {
    fn from(e: FeasibilityError) -> Self {
        Self::Feasibility(e)
    }
}

impl From<ConstraintViolationError> for ProblemLoaderError {
    fn from(e: ConstraintViolationError) -> Self {
        Self::Constraint(e)
    }
}

// ----------------------------------------------------------------
// Token Reader
// ----------------------------------------------------------------

/// A stateful helper that wraps a buffered reader and vends one
/// whitespace-delimited token at a time. Blank lines are silently skipped.
struct TokenReader<R> {
    reader: R,
    tokens: VecDeque<String>,
}

impl<R: BufRead> TokenReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            tokens: VecDeque::new(),
        }
    }

    /// Pulls the next raw string token from the stream.
    fn next_token_str(&mut self) -> Result<String, ProblemLoaderError> {
        while self.tokens.is_empty() {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line)?;

            if bytes_read == 0 {
                return Err(ProblemLoaderError::UnexpectedEof);
            }

            self.tokens
                .extend(line.split_whitespace().map(String::from));
        }

        Ok(self.tokens.pop_front().unwrap())
    }

    /// Reads the next token and parses it into the target type `T`.
    pub fn next_parsed<T: FromStr>(&mut self) -> Result<T, ProblemLoaderError> {
        let raw_str = self.next_token_str()?;
        raw_str.parse::<T>().map_err(|_| {
            ParseTokenError {
                token: raw_str,
                type_name: std::any::type_name::<T>(),
            }
            .into()
        })
    }
}

// ----------------------------------------------------------------
// Instance Loader
// ----------------------------------------------------------------

/// A configured loader for DBAP instances from the standard text format.
pub struct InstanceLoader<T> {
    pub forbidden_threshold: T,
}

impl<T> InstanceLoader<T>
where
    T: SolverNumeric + FromStr + PartialOrd + Display + Copy,
{
    /// Creates a new loader with the specified forbidden assignment threshold.
    pub fn new(forbidden_threshold: T) -> Self {
        Self {
            forbidden_threshold,
        }
    }

    /// Parses a DBAP instance from any readable IO stream.
    pub fn load_dbap<R: Read>(&self, reader: R) -> Result<Model<T>, ProblemLoaderError> {
        let mut tr = TokenReader::new(BufReader::new(reader));

        // 1. Problem Dimensions
        let num_vessels: usize = tr.next_parsed()?;
        let num_berths: usize = tr.next_parsed()?;

        if num_vessels == 0 || num_berths == 0 {
            return Err(ProblemLoaderError::InvalidDimensions);
        }

        // 2. Vessel Arrival Times
        let mut arrival_times = Vec::with_capacity(num_vessels);
        for _ in 0..num_vessels {
            arrival_times.push(tr.next_parsed::<T>()?);
        }

        // 3. Berth Opening Times
        let mut berth_starts = Vec::with_capacity(num_berths);
        for _ in 0..num_berths {
            berth_starts.push(tr.next_parsed::<T>()?);
        }

        // 4. Processing-time Matrix
        let mut processing_times = Vec::with_capacity(num_vessels * num_berths);
        for v in 0..num_vessels {
            let mut has_valid_berth = false;

            for _ in 0..num_berths {
                let raw: T = tr.next_parsed()?;
                if raw >= self.forbidden_threshold {
                    processing_times.push(ProcessingTime::none());
                } else {
                    has_valid_berth = true;
                    processing_times.push(ProcessingTime::some(raw));
                }
            }

            // Check Feasibility: A vessel must have at least one valid berth.
            if !has_valid_berth {
                return Err(FeasibilityError {
                    vessel_index: VesselIndex::new(v),
                }
                .into());
            }
        }

        // 5. Berth Closing Times
        let mut berth_ends = Vec::with_capacity(num_berths);
        for (b, &berth_start) in berth_starts.iter().enumerate().take(num_berths) {
            let end: T = tr.next_parsed()?;
            if end < berth_start {
                return Err(ProblemLoaderError::from(
                    ConstraintViolationError::BerthClosingBeforeOpening {
                        berth_index: BerthIndex::new(b),
                        opening_time: berth_start.to_string(),
                        closing_time: end.to_string(),
                    },
                ));
            }
            berth_ends.push(end);
        }

        // 6. Vessel Latest Departure Times
        let mut latest_departure_times = Vec::with_capacity(num_vessels);
        for (v, &arrival_time) in arrival_times.iter().enumerate().take(num_vessels) {
            let dep: T = tr.next_parsed()?;
            if dep < arrival_time {
                return Err(ConstraintViolationError::VesselDepartureBeforeArrival {
                    vessel_index: VesselIndex::new(v),
                    arrival_time: arrival_time.to_string(),
                    departure_time: dep.to_string(),
                }
                .into());
            }
            latest_departure_times.push(dep);
        }

        // 7. Assemble Derived Fields
        let unit_weight: T = "1"
            .parse()
            .map_err(|_| ConstraintViolationError::DefaultWeightParseFailed)?;
        let vessel_weights = vec![unit_weight; num_vessels];

        let mut opening_times = Vec::with_capacity(num_berths);
        for b in 0..num_berths {
            opening_times.push(vec![ClosedOpenInterval::new(
                berth_starts[b],
                berth_ends[b],
            )]);
        }

        Ok(Model::new(
            num_vessels,
            num_berths,
            arrival_times,
            latest_departure_times,
            vessel_weights,
            processing_times,
            opening_times,
        ))
    }

    /// Parses a DBAP instance from an in-memory string slice.
    #[allow(dead_code)]
    pub fn load_dbap_from_string(&self, data: &str) -> Result<Model<T>, ProblemLoaderError> {
        // `&[u8]` implements `Read`, bypassing any heavy stream wrappers
        self.load_dbap(data.as_bytes())
    }

    /// Reads a DBAP instance from the filesystem.
    pub fn load_dbap_file<P: AsRef<Path>>(&self, path: P) -> Result<Model<T>, ProblemLoaderError> {
        let file = File::open(path)?;
        self.load_dbap(file)
    }
}
