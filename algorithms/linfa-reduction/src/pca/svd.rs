use ndarray::{Array1, Array2, Axis};
use linfa::Float;
use std::ops::DivAssign;
use ndarray_linalg::{generate, TruncatedOrder, Lapack};
use num_traits::NumCast;
use ndarray_linalg::lobpcg::{lobpcg, LobpcgResult};
use linfa::linalg::error::LinalgError;


pub trait MagnitudeCorrection {
    fn correction() -> Self;
}

impl MagnitudeCorrection for f32 {
    fn correction() -> Self {
        1.0e3
    }
}

impl MagnitudeCorrection for f64 {
    fn correction() -> Self {
        1.0e6
    }
}

/// The result of a eigenvalue decomposition, not yet transformed into singular values/vectors
///
/// Provides methods for either calculating just the singular values with reduced cost or the
/// vectors with additional cost of matrix multiplication.
#[derive(Debug)]
pub struct TruncatedSvdResult<A> {
    eigvals: Array1<A>,
    eigvecs: Array2<A>,
    problem: Array2<A>,
    ngm: bool,
}

pub fn svd<F: Float + Lapack>(problem: Array2<F>, b:&Array2<F>, n_components: usize, precision: f32, max_iterations: usize) -> Result<TruncatedSvdResult<F>, LinalgError> {
    if n_components < 1 {
        panic!("The number of singular values to compute should be larger than zero!");
    }

    let (n, m) = (problem.nrows(), problem.ncols());

    // generate initial matrix
    let x: Array2<f32> = generate::random((usize::min(n, m), n_components));
    let x = x.mapv(|x| NumCast::from(x).unwrap());

    // square precision because the SVD squares the eigenvalue as well
    let precision = precision * precision;

    // use problem definition with less operations required
    let res = if n > m {
        lobpcg(
            |y| problem.t().dot(&problem.dot(&b.dot(&y))),
            x,
            |_| {},
            None,
            precision,
            max_iterations,
            TruncatedOrder::Largest,
        )
    } else {
        lobpcg(
            |y| b.t().dot(&problem.dot(&problem.t().dot(&y))),
            x,
            |_| {},
            None,
            precision,
            max_iterations,
            TruncatedOrder::Largest,
        )
    };

    // convert into TruncatedSvdResult
    match res {
        LobpcgResult::Ok(vals, vecs, _) | LobpcgResult::Err(vals, vecs, _, _) => {
            Ok(TruncatedSvdResult {
                problem,
                eigvals: vals,
                eigvecs: vecs,
                ngm: n > m,
            })
        }
        LobpcgResult::NoResult(err) => Err(err),
    }
}


impl<A: Float + PartialOrd + DivAssign<A> + 'static + MagnitudeCorrection> TruncatedSvdResult<A> {
    /// Returns singular values ordered by magnitude with indices.
    fn singular_values_with_indices(&self) -> (Array1<A>, Vec<usize>) {
        // numerate eigenvalues
        let mut a = self.eigvals.iter().enumerate().collect::<Vec<_>>();

        // sort by magnitude
        a.sort_by(|(_, x), (_, y)| x.partial_cmp(&y).unwrap().reverse());

        // calculate cut-off magnitude (borrowed from scipy)
        let cutoff = A::epsilon() * // float precision
            A::correction() * // correction term (see trait below)
            *a[0].1; // max eigenvalue

        // filter low singular values away
        let (values, indices): (Vec<A>, Vec<usize>) = a
            .into_iter()
            .filter(|(_, x)| *x > &cutoff)
            .map(|(a, b)| (b.sqrt(), a))
            .unzip();

        (Array1::from(values), indices)
    }

    /// Returns singular values ordered by magnitude
    pub fn values(&self) -> Array1<A> {
        let (values, _) = self.singular_values_with_indices();

        values
    }

    /// Returns singular values, left-singular vectors and right-singular vectors
    pub fn values_vectors(&self) -> (Array2<A>, Array1<A>, Array2<A>) {
        let (values, indices) = self.singular_values_with_indices();

        // branch n > m (for A is [n x m])
        let (u, v) = if self.ngm {
            let vlarge = self.eigvecs.select(Axis(1), &indices);
            let mut ularge = self.problem.dot(&vlarge);

            ularge
                .gencolumns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        } else {
            let ularge = self.eigvecs.select(Axis(1), &indices);

            let mut vlarge = self.problem.t().dot(&ularge);
            vlarge
                .gencolumns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        };

        (u, values, v.reversed_axes())
    }
}
