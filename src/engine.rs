use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

#[derive(Clone)]
pub enum Op {
    Add,
    Mul,
    Exp,
    Pow,
    ReLU,
    TanH,
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Op::Add => { write!(f, "+") }
            Op::Mul => { write!(f, "*") }
            Op::Exp => { write!(f, "exp") }
            Op::Pow => { write!(f, "pow") }
            Op::ReLU => { write!(f, "relu") }
            Op::TanH => { write!(f, "tanh") }
        }
    }
}

struct InnerValue {
    data: f64,
    grad: f64,
    prev: Vec<Value>,
    op: Option<Op>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<InnerValue>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data,
            grad: 0.0,
            prev: Vec::new(),
            op: None,
        })))
    }

    pub fn exp(&self) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data: self.get_data().exp(),
            grad: 0.0,
            prev: vec![self.clone()],
            op: Some(Op::Exp),
        })))
    }

    pub fn pow(&self, other: f64) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data: self.get_data().powf(other),
            grad: 0.0,
            prev: vec![self.clone(), Value::new(other)],
            op: Some(Op::Pow),
        })))
    }

    pub fn relu(&self) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data: if self.get_data() < 0.0 { 0.0 } else { self.get_data() },
            grad: 0.0,
            prev: vec![self.clone()],
            op: Some(Op::ReLU),
        })))
    }

    pub fn tanh(&self) -> Self {
        let x = self.get_data();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Self(Rc::new(RefCell::new(InnerValue {
            data: t,
            grad: 0.0,
            prev: vec![self.clone()],
            op: Some(Op::TanH),
        })))
    }

    pub fn get_grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    fn add_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad;
    }

    pub fn get_data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn get_op(&self) -> Option<Op> { self.0.borrow().op.clone() }

    fn lchild(&self) -> Self {
        self.0.borrow().prev[0].clone()
    }

    fn rchild(&self) -> Self {
        self.0.borrow().prev[1].clone()
    }

    pub fn get_children(&self) -> Vec<Self> { self.0.borrow().prev.clone() }

    pub fn backward(&self) {
        fn backward_prev(value: &Value) {
            // Calculate grad for value children in prev (non-recursively).
            // This match is equivalent to the original _backward() method.
            match value.0.borrow().op {
                // a = b + c
                //   b.grad += 1.0 * a.grad
                //   c.grad += 1.0 * a.grad
                Some(Op::Add) => {
                    value.lchild().add_grad(value.get_grad());
                    value.rchild().add_grad(value.get_grad());
                }
                // a = b * c
                //   b.grad += c.data * a.grad
                //   c.grad += b.data * a.grad
                Some(Op::Mul) => {
                    value.lchild().add_grad(value.rchild().get_data() * value.get_grad());
                    value.rchild().add_grad(value.lchild().get_data() * value.get_grad());
                }
                // a = b.exp()
                //   b.grad += a.data * a.grad
                Some(Op::Exp) => {
                    value.lchild().add_grad(value.get_data() * value.get_grad());
                }
                // a = b.pow(x)
                //   b.grad += x * (b.data^(x - 1)) * a.grad
                Some(Op::Pow) => {
                    let x = value.rchild().get_data();
                    let b_data = value.lchild().get_data();
                    value.lchild().add_grad(x * (b_data.powf(x - 1.0)) * value.get_grad());
                }
                // a = b.relu()
                //   b.grad += a.data > 0 ? a.grad : 0
                Some(Op::ReLU) => {
                    value.lchild().add_grad(if value.get_data() > 0.0 { value.get_grad() } else { 0.0 });
                }
                Some(Op::TanH) => {
                    let x = value.lchild().get_data();
                    let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                    value.lchild().add_grad((1.0 - t.powi(2)) * value.get_grad());
                }
                None => {}
            }
        }

        fn build_topo(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                for child in &v.0.borrow().prev {
                    build_topo(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        let mut topo: Vec<Value> = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut topo, &mut visited);
        self.set_grad(1.0);
        for node in topo.iter().rev() {
            backward_prev(node);
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        Display::fmt(&self, f)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Value(data={:#?}, grad={:#?} )", self.get_data(), self.get_grad())
    }
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        //self.0.as_ptr().hash(state)
        Rc::as_ptr(&self.0).hash(state)
    }
}

// x + y
impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Self(Rc::new(RefCell::new(InnerValue {
            data: self.get_data() + other.get_data(),
            grad: 0.0,
            prev: vec![self, other],
            op: Some(Op::Add),
        })))
    }
}

// x + 1.0
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + Value::new(other)
    }
}

// 1.0 + x
impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value::new(self) + other
    }
}

// x += y
impl AddAssign for Value {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

// &x + &y
impl Add for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        self.clone() + other.clone()
    }
}

// &x + 1.0
impl Add<f64> for &Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + &Value::new(other)
    }
}

// 1.0 + &x
impl Add<&Value> for f64 {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        other + self
    }
}

// x * y
impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(InnerValue {
            data: self.get_data() * other.get_data(),
            grad: 0.0,
            prev: vec![self, other],
            op: Some(Op::Mul),
        })))
    }
}

// x * 1.0
impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * Value::new(other)
    }
}

// 1.0 * x
impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        other * self
    }
}

// x *= y
impl MulAssign for Value {
    fn mul_assign(&mut self, other: Value) {
        self.set_data(self.get_data() * other.get_data());
    }
}

// &x * &y
impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: Self) -> Value {
        self.clone() * other.clone()
    }
}

// &x * 1.0
impl Mul<f64> for &Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * &Value::new(other)
    }
}

// 1.0 * &x
impl Mul<&Value> for f64 {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        other * self
    }
}

// x / y
impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.pow(-1.0)
    }
}

// x / 1.0
impl Div<f64> for Value {
    type Output = Value;
    fn div(self, other: f64) -> Value {
        self * other.powf(-1.0)
    }
}

// 1.0 / x
impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.pow(-1.0)
    }
}

// x /= y
impl DivAssign for Value {
    fn div_assign(&mut self, other: Self) {
        self.set_data(self.get_data() / other.get_data());
    }
}

// &x / &y
impl Div for &Value {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        self * &other.pow(-1.0)
    }
}

// &x / 1.0
impl Div<f64> for &Value {
    type Output = Value;
    fn div(self, other: f64) -> Value {
        self * other.powf(-1.0)
    }
}

// 1.0 / &x
impl Div<&Value> for f64 {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        self * other.pow(-1.0)
    }
}

// -x
impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}

// -&x
impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}

// x - y
impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

// x - 1.0
impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, other: f64) -> Value {
        self + (-other)
    }
}

// 1.0 - x
impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

// x -= y
impl SubAssign for Value {
    fn sub_assign(&mut self, other: Self) {
        self.set_data(self.get_data() - other.get_data());
    }
}

// &x - &y
impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

// &x - 1.0
impl Sub<f64> for &Value {
    type Output = Value;
    fn sub(self, other: f64) -> Value {
        self + (-other)
    }
}

// 1.0 - &x
impl Sub<&Value> for f64 {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self + (-other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_twice() {
        let a = Value::new(3.0);
        let b = &a + &a;
        b.backward();
        assert_eq!(3.0, a.get_data());
        assert_eq!(2.0, a.get_grad());
        assert_eq!(6.0, b.get_data());
        assert_eq!(1.0, b.get_grad());
    }

    #[test]
    fn add_and_multiply() {
        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        let d = &a + &b;
        let e = &c * &d;
        e.backward();
        assert_eq!(-2.0, a.get_data());
        assert_eq!(-3.0, a.get_grad());
        assert_eq!(3.0, b.get_data());
        assert_eq!(-8.0, b.get_grad());
        assert_eq!(-6.0, c.get_data());
        assert_eq!(1.0, c.get_grad());
        assert_eq!(1.0, d.get_data());
        assert_eq!(-6.0, d.get_grad());
        assert_eq!(-6.0, e.get_data());
        assert_eq!(1.0, e.get_grad());
    }

    /*
    https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/test/test_engine.py
    import torch
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    # x.grad.item() = 46.0
    # y.data.item() = -20.0
    */
    #[test]
    fn pytorch_sanity() {
        const XPT_GRAD: f64 = 46.0;
        const YPT_DATA: f64 = -20.0;

        let x = Value::new(-4.0);
        let z = 2.0 * &x + (2.0 + &x);
        let q = z.relu() + &z * &x;
        let h = (&z * &z).relu();
        let y = &h + &q + &q * &x;
        y.backward();

        assert_eq!(XPT_GRAD, x.get_grad());
        assert_eq!(YPT_DATA, y.get_data());
    }

    /*
    https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/test/test_engine.py
    import torch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    # a.grad.item() = 138.83381924198252
    # b.grad.item() = 645.5772594752186
    # g.data.item() = 24.70408163265306
    */
    #[test]
    fn pytorch_sanity_more() {
        const APT_GRAD: f64 = 138.83381924198252;   // The numerical value of dg/da
        const BPT_GRAD: f64 = 645.5772594752186;    // The numerical value of dg/db
        const GPT_DATA: f64 = 24.70408163265306;    // The outcome of this forward pass
        const TOLERANCE: f64 = 1e-6;

        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let mut c = &a + &b;
        let mut d = &(&a * &b) + &b.pow(3.0);
        c += &c + 1.0;
        c += 1.0 + &c + (-&a);
        d += &d * 2.0 + (&b + &a).relu();
        d += 3.0 * &d + (&b - &a).relu();
        let e = c - d;
        let f = e.pow(2.0);
        let mut g = &f / 2.0;
        g += 10.0 / f;
        g.backward();

        // Backward pass went well.
        assert!((a.get_grad() - APT_GRAD).abs() < TOLERANCE);
        assert!((b.get_grad() - BPT_GRAD).abs() < TOLERANCE);
        // Forward pass went well.
        assert!((g.get_data() - GPT_DATA).abs() < TOLERANCE);
    }
}
