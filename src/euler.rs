// Solvers for the ode y'(t) = βy(t) with intial condition y(0) = y₀

const Y0  : f64 = 1.0;
const T0  : f64 = 0.0;


#[derive(Debug,Copy, Clone)]
struct SolutionElement{
    time : f64,
    val  : f64,
}

struct SolutionSequence{
    stepsize  : f64,
    beta      : f64,
    soln_elem : SolutionElement,
}

const INITIAL_CONDITION: SolutionElement = SolutionElement{time: T0, val: Y0};

impl SolutionSequence{
   fn new(stepsize : f64, beta: f64) -> SolutionSequence{
       SolutionSequence{stepsize, 
                        beta, 
                        soln_elem: INITIAL_CONDITION}
   }
}

fn rk2_next(current: SolutionElement, beta: f64, stepsize: f64) -> SolutionElement{

    let deriv = |y| beta*y;

    let t0 = current.time;
    let y0 = current.val;

    let d1 = deriv(y0);
    let k1 = d1;


    let d2 = k1;
    let k2 = deriv(y0 + stepsize*d2);
    
    let dy = 0.5*(k1 + k2);

    let t1 = t0 + stepsize;
    let y1 = y0 + stepsize*dy;

    SolutionElement{ time: t1, val: y1}
}


impl Iterator for SolutionSequence{

    type Item = SolutionElement;

    fn next(&mut self) -> Option<Self::Item>{
       let next_soln_elem = rk2_next(self.soln_elem, self.beta, self.stepsize);

       self.soln_elem =  next_soln_elem;
  
       Some(next_soln_elem)
    }
}
        
   

pub fn rk2(beta: f64, stepsize: f64, finish_time :f64) -> f64{
    let mut soln_seq = SolutionSequence::new(stepsize, beta);

    let n = (finish_time/ stepsize) as usize;
  
    soln_seq.nth(n).unwrap().val

}



