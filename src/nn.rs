use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::iter::zip;
use rand::distributions::{Distribution, Uniform};
use crate::engine::Value;

#[derive(Clone, Copy)]
pub enum ActFunc {
    ReLU,
    TanH,
}

pub trait Module {
    fn parameters(&self) -> Vec<Value>;
    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }
}

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
    act: ActFunc,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool, act: ActFunc) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::new_inclusive(-1.0, 1.0);
        Self {
            weights: between.sample_iter(&mut rng).take(nin).map(Value::new).collect(),
            bias: Value::new(between.sample(&mut rng)),
            nonlin,
            act,
        }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Value {
        // sum(weights * inputs + bias)
        let act = zip(self.weights.iter(), inputs.iter())
            .map(|(wi, xi)| wi * xi)
            .fold(self.bias.clone(), |a, b| a + b);

        match self.act {
            ActFunc::TanH => { act.tanh() }
            ActFunc::ReLU => {
                if self.nonlin { act.relu() } else { act }
            }
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        [self.weights.clone(), vec![self.bias.clone()]].concat()
    }
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}Neuron({})", if self.nonlin { "ReLU" } else { "Linear" }, self.weights.len())
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: usize, nout: usize, nonlin: bool, act: ActFunc) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin, act)).collect()
        }
    }

    fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(inputs.clone())).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters: Vec<Value> = Vec::new();
        for neuron in self.neurons.iter() {
            parameters.extend(neuron.parameters());
        }
        parameters
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Layer of [{:?}]", self.neurons)
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize], act: ActFunc) -> Self {
        let mut sz = vec![nin];
        sz.extend_from_slice(nouts);
        Self {
            layers: (0..nouts.len())
                .map(|i| Layer::new(sz[i], sz[i + 1], i != nouts.len() - 1, act))
                .collect(),
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        // Convert array of f64 inputs to vector of Value.
        //let mut v: Vec<Value> = x.iter().map(|x| Value::new(*x)).collect();
        let mut v = x.clone();
        for layer in self.layers.iter() {
            v = layer.forward(v);
        }
        v
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters: Vec<Value> = Vec::new();
        for layer in self.layers.iter() {
            parameters.extend(layer.parameters());
        }
        parameters
    }
}

impl Debug for MLP {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "MLP of [{:?}]", self.layers)
    }
}
