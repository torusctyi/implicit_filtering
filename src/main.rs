mod euler;
use euler::rk2;

const BETA       : f64 = 1.0;
const FINAL_TIME : f64 = 5.0; 

fn main() {
    
    let result = implicit_filtering::implicit_filtering(get_mse_rk2, 1.5, 0.1, 0.0000001);

    println!("\nFinal Result: β = {0: <+12.10}, MSE = {1: <+12.10}", result.x, result.mse);
}

fn get_mse_rk2(x: f64, h:f64) -> f64{
    let true_val = (BETA*FINAL_TIME).exp();
    let estimated_val = rk2(x,h, FINAL_TIME);

    let error = true_val - estimated_val;

    error.powi(2)
}


