const std = @import("std");
const assert = std.debug.assert;

pub fn RangeTo(n: usize) type {
    return struct {
        idx: usize,
        pub fn new() RangeTo(n) {
            return .{ .idx = 0 };
        }
        pub fn next(self: *RangeTo(n)) ?usize {
            if (self.idx >= n) {
                return null;
            }
            var r = self.idx;
            self.idx += 1;
            return r;
        }
    };
}

pub const Relu1 = struct {
    pub const Param = struct {};
    pub const Input = f32;
    pub const Output = f32;

    fn initializeParams(param: *Param, rng: *std.rand.Random) void {
        // nothing
    }
    fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
        // nothing
    }

    fn run(input: *Input, param: Param, output: *Output) void {
        if (input > 0) {
            output.* = input;
        }
    }
    fn reverse(
        input: Input,
        param: Param,
        delta: Output,
        backprop: *Input,
        gradient: *Param,
    ) void {
        if (input > 0) {
            backprop.* = delta;
        }
    }
};

pub fn Lin(size: usize) type {
    return struct {
        pub const Param = struct { weights: [size]f32, bias: f32 };
        pub const Input = [size]f32;
        pub const Output = f32;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            var iter = RangeTo(size).new();
            while (iter.next()) |i| {
                param.weights[i] = rng.float(f32) * 2 - 1;
            }
            param.bias = rng.float(f32) * 4 - 2;
        }
        fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
            var iter = RangeTo(size).new();
            while (iter.next()) |i| {
                gradient.weights[i] += scale * update.weights[i];
            }
            gradient.bias += scale * update.bias;
        }

        fn run(input: Input, param: Param, output: *Output) void {
            for (input) |x, i| {
                output.* += x * param.weights[i];
            }
            output.* += param.bias;
        }
        fn reverse(
            input: Input,
            param: Param,
            delta: Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            gradient.bias = delta;
            for (input) |x, i| {
                backprop.*[i] += delta * param.weights[i];
                gradient.weights[i] += delta * x;
            }
        }
    };
}

pub fn zeroed(comptime T: type) T {
    var x: T = undefined;
    @memset(@ptrCast([*]u8, &x), 0, @sizeOf(T));
    return x;
}

fn Seq2(comptime Net1: type, comptime Net2: type) type {
    assert(Net1.Output == Net2.Input);

    return struct {
        pub const Param = struct {
            first: Net1.Param,
            second: Net2.Param,
        };
        pub const Input = Net1.Input;
        pub const Output = Net2.Output;

        pub fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            Net1.initializeParams(&param.first, rng);
            Net2.initializeParams(&param.second, rng);
        }
        pub fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
            Net1.updateGradient(&gradient.first, scale, &update.first);
            Net2.updateGradient(&gradient.second, scale, &update.second);
        }

        pub fn run(input: Input, param: Param, output: *Output) void {
            var scratch: Net1.Output = zeroed(Net1.Output);
            Net1.run(input, &param.first, &scratch);
            Net2.run(scratch, &param.second, output);
        }
        pub fn reverse(
            input: Input,
            param: Param,
            delta: Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            var scratch: Net1.Output = zeroed(Net1.Output);
            Net1.run(input, &param.first, &scratch);
            var middleDelta: Net1.Output = zeroed(Net1.Output);
            Net2.reverse(scratch, &param.second, delta, &middleDelta, &gradient.second);
            Net1.reverse(input, &param.first, middleDelta, backprop, &gradient.first);
        }
    };
}

fn SeqFrom(comptime Nets: var, index: usize) type {
    if (index == Nets.len - 1) {
        return Nets[index];
    }
    return Seq2(Nets[index], SeqFrom(Nets, index + 1));
}

pub fn Seq(comptime Nets: var) type {
    assert(Nets.len > 0);
    return SeqFrom(Nets, 0);
}

pub fn Fan(comptime by: usize, Net: type) type {
    return struct {
        pub const Input = Net.Input;
        pub const Output = [by]Net.Output;
        pub const Param = [by]Net.Param;

        pub fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.initializeParams(&param[i], rng);
            }
        }
        pub fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.updateGradient(&gradient.*[i], scale, &update.*[i]);
            }
        }

        pub fn run(input: Input, param: Param, output: *Output) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.run(input, param[i], &output[i]);
            }
        }
        pub fn reverse(
            input: Input,
            param: Param,
            delta: Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.reverse(input, param[i], delta[i], backprop, &gradient[i]);
            }
        }
    };
}

pub fn Relu(comptime in: usize, comptime out: usize) type {
    return Fan(out, Seq(.{ Lin(in), Relu1 }));
}

pub fn LossSum(comptime LossNet: type) type {
    assert(LossNet.Output == f32);
    return struct {
        pub const Input = []LossNet.Input;
        pub const Output = f32;
        pub const Param = LossNet.Param;

        pub fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            LossNet.initializeParams(param, rng);
        }
        pub fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
            LossNet.updateGradient(gradient, scale, update);
        }

        pub fn run(input: Input, param: Param, output: *Output) void {
            var loss: f32 = 0.0;

            for (input) |item, index| {
                var lossAdd: f32 = 0.0;
                LossNet.run(input[index], param, &lossAdd);
                loss += lossAdd;
            }

            output.* += loss;
        }
        pub fn reverse(
            input: Input,
            param: Param,
            delta: Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            // TODO: backprop is not set; should we have non-differentiable inputs?
            for (input) |_, index| {
                var discardInputDelta = zeroed(LossNet.Input);
                // here we rely on gradients being added, instead of set:
                LossNet.reverse(input[index], param, delta, &discardInputDelta, gradient);
            }
        }
    };
}

pub fn TrainingExample(comptime T: type, comptime G: type) type {
    return struct {
        input: T,
        target: G,
    };
}

pub fn LossL2(comptime Net: type) type {
    return struct {
        pub const Input = TrainingExample(Net.Input, f32);
        pub const Output = f32;
        pub const Param = Net.Param;

        pub fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            Net.initializeParams(param, rng);
        }
        pub fn updateGradient(gradient: *Param, scale: f32, update: *Param) void {
            Net.updateGradient(gradient, scale, update);
        }

        pub fn run(input: Input, param: Param, output: *Output) void {
            var predicted = [1]f32{0};
            Net.run(input.input, param, &predicted);
            var loss = (predicted[0] - input.target) * (predicted[0] - input.target);
            output.* += loss;
        }
        pub fn reverse(
            input: Input,
            param: Param,
            delta: Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            // 'delta' indicates how much being wrong counts.
            // the amount we pass back into the previous layer is therefore based
            // on how far off the current estimate is.

            // So we first need to run forward to obtain a prediction:
            var predicted = [1]f32{0};
            Net.run(input.input, param, &predicted);

            // We have: L = (pred - target)^2
            // and we know dE / dL
            // we want to find dpred / dL

            // dE/dA = dE/dL dL/dA
            // so take d/dpred of both sides:
            // dL/dpred = 2(pred-target)

            // so dE/dpred = dE/dL 2 (pred - target).

            var adjustedDelta: Net.Output = [1]f32{2 * delta * (predicted[0] - input.target)};
            var discardInputBackprop = zeroed(Net.Input);
            Net.reverse(
                input.input,
                param,
                adjustedDelta,
                &discardInputBackprop,
                gradient,
            );
        }
    };
}
