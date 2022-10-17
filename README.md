# micrograd_rs

Rust implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
for purposes of learning both ML and Rust.

## Main takeaways

Basically the same takeaways that everybody goes through it seems.

- ML is not easy, but nothing magical eigher :)
- Writing the same thing in Rust takes multiple times the effort than in Python
(at least at the learning stage).
- Analyzing Rust code is multiple times easier than Python because of the strict
static typing, and explicitness. I'm not talking about the general understanding
of what the code does, but understanding what's really going on.
- Rust compiler error messages and hints are enough to understand the problem in
most situations.
- Rust is **much** faster than Python (which isn't surprising, so is C), and it
is an enormous advantage for ML, as training on huge datasets can take a lot of
time. For example, on my machine the Python
[micrograd demo](https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/demo.ipynb)
optimization loop took around **82 seconds**, while Rust implementation only
around **1.5 seconds** in release mode 🤯 And around 20 seconds in debug,
so don't forget to build your crates with `--release`, kids.

## Demo

See the provided `.ipynb` notebooks for [petgraph](https://github.com/petgraph/petgraph)
visualizations, and example training with usage of TanH and ReLU functions.

The TanH example is from Karpathy's
["Neural Networks: Zero to Hero" notes](https://github.com/karpathy/nn-zero-to-hero/blob/44cfac7e03782b78f6bf2390d37926453295c90a/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb), and the ReLU example is from [micrograd demo notebook](https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/demo.ipynb).

In order to run the notebooks locally, add Rust Jupyter Kernel with
[evcxr_jupyter](https://crates.io/crates/evcxr_jupyter).

```sh
rustup component add rust-src
cargo install evcxr_jupyter
~/.cargo/bin/evcxr_jupyter --install
jupyter notebook
```

## Value operations

Operations with both owned and borrowed `Value` are supported, as well as with
`f64`. The `pow`, `exp`, `tanh`, and `relu` functions are also supported.

```rust
use micrograd_rs::engine::Value;

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

println!("{:.4}", g.get_data());  // 24.7041, the outcome forward pass.
println!("{:.4}", a.get_grad());  // 138.8338, the numerical value of dg/da.
println!("{:.4}", b.get_grad());  // 645.5773, the numerical value of dg/db.
```
