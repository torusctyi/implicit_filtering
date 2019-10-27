
const LINE_SEARCH_REDUCTION: f64 = 0.7;
const STENCIL_REDUCTION: f64 = 0.25;
const ARMIJO_CONSTANT: f64 = 0.001;
const MAX_ITERS: usize = 10;

#[derive(Clone)]
#[derive(Copy)]
#[derive(PartialEq)]
pub struct OptimResult{
   pub x: f64,
   pub mse: f64,
}

fn report_stencil_failure( msg: &str){
    eprintln!("\nStencil Failure: {}", msg); 
}

// estimate the gradient of the objective function
fn generate_gradient(mse: fn(f64, f64) -> f64, result:  &OptimResult, h: f64) -> Option<(f64, f64)>{

   let mse_centre   = result.mse;
   let mse_right = mse(result.x + h,h);
   let mse_left  = mse(result.x - h,h);


   let grad = (mse_right - mse_left)/(2.0*h);
   let hess = (mse_right + mse_left - 2.0*mse_centre)/(h*h);

   // if the centre point is the smallest so that no descent direction can be identified, or if 
   // the first derivative is estimated to be small relative to the stepsize, report an error
   let no_descent_direction = mse_right >= mse_centre &&  mse_left >= mse_centre;
   let grad_o_h = grad.abs() <= h;

   if no_descent_direction || grad_o_h{ 
        None
   } else {
        Some((grad, hess))
   }
}

// A backtracking line search that attempts to find a point that satisfies the 
// Armijo Condition. Since only an approximate gradient is used, this search is not guaranteed to a
// actually succeed
fn backtracking_line_search(mse: fn(f64, f64) -> f64, x: f64, p: f64, grad: f64, h:f64) -> Option<OptimResult> 
{
    let mse_old  = mse(x,h);

    for i in 0..MAX_ITERS{

        let a = LINE_SEARCH_REDUCTION.powi(i as i32); 

        let x_new            = x + a*p;
        let mse_new          = mse(x_new, h);

        let required_decrease =  ARMIJO_CONSTANT*a*p*grad;
        let actual_decrease = mse_new - mse_old;

        if actual_decrease <= required_decrease{
            return Some(OptimResult{x: x_new, mse: mse_new}) 
        }
    }

    None
}

// A line search algorithm that approximately computes the gradient and Hessian using
// finite differences
fn grad_search(mse: fn(f64, f64) -> f64, x: f64, h: f64) -> Option<OptimResult>{

    let old_result = OptimResult{ x, mse: mse(x,h)};

    let mut current_result = old_result;

    eprintln!("\nCommencing optimisation routine:\n   h = {0: <12}\n   β = {1: <12}\n", h, x);

    eprintln!("{0: ^+013.10}|{1: ^018.10}|{2: ^019.10}|", "   β", "MSE", "‖∇ₕMSE‖");
    eprintln!("==============================================================");
    
    for _i in 0..MAX_ITERS{

        // attempt to compute approximate gradient and Hessian
        let (grad, hess) = match generate_gradient(mse, &current_result, h){
                       Some(gh)   => gh,
                       None       => { eprintln!("{0: ^+013.10}|{1: ^018.10}|{2: ^019.10}|", 
                                                    current_result.x, current_result.mse, "N/A");
                                       report_stencil_failure("Unable to clearly estimate gradient");
                                       break},
                };
                        
        // compute quasi-Newton search direction          
        let p  = -grad.signum()*grad.abs()/hess;

        let p = if p*grad <= 0.0 {p} else {-grad.signum()*grad.abs()}; // check that a descent direction is defined
        let p = if p.abs() <= 3.0 {p} else {-grad.signum()*3.0};       // check the search direction isn't too big

  

        // print table row
        eprintln!("{0: ^+013.10}|{1: ^018.10}|{2: ^019.10}|", current_result.x, current_result.mse, grad.abs());

        assert!(p*grad <= 0.0); // this should always be true, but check anyway just in case

        // conduct a backtracking line search
        match backtracking_line_search(mse, current_result.x, p, grad, h){
            Some(result) => current_result = result,
            None         => {report_stencil_failure("Line Search Failure");
                             break;},
        };

    }

    if current_result == old_result || current_result.mse >= old_result.mse{
        None
    } else {
        Some(current_result)
    }
}

pub fn implicit_filtering(mse: fn(f64, f64) -> f64, x0: f64, h0: f64, tol: f64) -> OptimResult{

    let mut old_result = OptimResult{x: x0, mse: mse(x0,h0)};

    for i in 0..20{
        let h :f64 = h0*STENCIL_REDUCTION.powi(i as i32);
        
        let grad_result =  grad_search(mse, old_result.x, h);

        let new_result = match grad_result{
                           Some(result) => result,
                           None         => continue
                        };

        let diff = (old_result.x - new_result.x).abs();

        old_result = new_result;

        // terminate when reducing the stepsize makes no difference
        if diff <= tol {
            break;
        }
    }

    old_result
}












        
        
            
        





    





