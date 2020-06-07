const std = @import("std");
const assert = std.debug.assert;

fn RangeTo(n: usize) type {
    return struct {
        idx: usize,
        fn new() RangeTo(n) {
            return .{ .idx = 0 };
        }
        fn next(self: *RangeTo(n)) ?usize {
            if (self.idx >= n) {
                return null;
            }
            var r = self.idx;
            self.idx += 1;
            return r;
        }
    };
}

const Relu1 = struct {
    const Param = struct {};
    const Input = f32;
    const Output = f32;

    fn initializeParams(param: *Param, rng: *std.rand.Random) void {
        // nothing
    }

    fn run(input: *Input, param: *Param, output: *Output) void {
        if (input.* > 0) {
            output.* = input.*;
        }
    }
    fn reverse(
        input: *Input,
        param: *Param,
        delta: *Output,
        backprop: *Input,
        gradient: *Param,
    ) void {
        if (input.* > 0) {
            backprop.* = delta.*;
        }
    }
};

fn Lin(size: usize) type {
    return struct {
        const Param = struct { weights: [size]f32, bias: f32 };
        const Input = [size]f32;
        const Output = f32;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            var iter = RangeTo(size).new();
            while (iter.next()) |i| {
                param.weights[i] = rng.float(f32) * 2 - 1;
            }
            param.bias = rng.float(f32) * 4 - 2;
        }

        fn run(input: *Input, param: *Param, output: *Output) void {
            for (input.*) |x, i| {
                output.* += x * param.weights[i];
            }
            output.* += param.bias;
        }
        fn reverse(
            input: *Input,
            param: *Param,
            delta: *Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            gradient.bias = delta.*;
            for (input.*) |x, i| {
                backprop.*[i] += delta.* * param.weights[i];
                gradient.weights[i] += delta.* * x;
            }
        }
    };
}

fn zeroed(comptime T: type) T {
    var x: T = undefined;
    @memset(@ptrCast([*]u8, &x), 0, @sizeOf(T));
    return x;
}

fn isTrivial(comptime t: type) bool {
    if (t == struct {}) {
        return true;
    }
    return false;
}

fn Seq2(comptime Net1: type, comptime Net2: type) type {
    assert(Net1.Output == Net2.Input);

    return struct {
        const Param = struct {
            first: Net1.Param,
            second: Net2.Param,
        };
        const Input = Net1.Input;
        const Output = Net2.Output;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            Net1.initializeParams(&param.first, rng);
            Net2.initializeParams(&param.second, rng);
        }

        fn run(input: *Input, param: *Param, output: *Output) void {
            var scratch: Net1.Output = zeroed(Net1.Output);
            Net1.run(input, &param.first, &scratch);
            Net2.run(&scratch, &param.second, output);
        }
        fn reverse(
            input: *Input,
            param: *Param,
            delta: *Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            var scratch: Net1.Output = zeroed(Net1.Output);
            Net1.run(input, &param.first, &scratch);
            var middleDelta: Net1.Output = zeroed(Net1.Output);
            Net2.reverse(&scratch, &param.second, delta, &middleDelta, &gradient.second);
            Net1.reverse(input, &param.first, &middleDelta, backprop, &gradient.first);
        }
    };
}

fn SeqFrom(comptime Nets: var, index: usize) type {
    if (index == Nets.len - 1) {
        return Nets[index];
    }
    return Seq2(Nets[index], SeqFrom(Nets, index + 1));
}

fn Seq(comptime Nets: var) type {
    assert(Nets.len > 0);
    return SeqFrom(Nets, 0);
}

fn Fan(comptime by: usize, Net: type) type {
    return struct {
        const Input = Net.Input;
        const Output = [by]Net.Output;
        const Param = [by]Net.Param;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.initializeParams(&param[i], rng);
            }
        }

        fn run(input: *Input, param: *Param, output: *Output) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.run(input, &param[i], &output[i]);
            }
        }
        fn reverse(
            input: *Input,
            param: *Param,
            delta: *Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            var iter = RangeTo(by).new();
            while (iter.next()) |i| {
                Net.reverse(input, &param[i], &delta[i], backprop, &gradient[i]);
            }
        }
    };
}

fn Relu(comptime in: usize, comptime out: usize) type {
    return Fan(out, Seq(.{ Lin(in), Relu1 }));
}

fn LossSum(comptime LossNet: type) type {
    assert(LossNet.Output == f32);
    return struct {
        const Input = []LossNet.Input;
        const Output = f32;
        const Param = LossNet.Param;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            LossNet.initializeParams(param, rng);
        }

        fn run(input: *Input, param: *Param, output: *Output) void {
            var loss: f32 = 0.0;

            for (input.*) |item, index| {
                var lossAdd: f32 = 0.0;
                LossNet.run(&input.*[index], param, &lossAdd);
                loss += lossAdd;
            }

            output.* += loss;
        }
        fn reverse(
            input: *Input,
            param: *Param,
            delta: *Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            // TODO: backprop is not set; should we have non-differentiable inputs?
            for (input.*) |_, index| {
                var discardInputDelta = zeroed(LossNet.Input);
                // here we rely on gradients being added, instead of set:
                LossNet.reverse(&input.*[index], param, delta, &discardInputDelta, gradient);
            }
        }
    };
}

fn TrainingExample(comptime T: type, comptime G: type) type {
    return struct {
        input: T,
        target: G,
    };
}

fn LossL2(comptime Net: type) type {
    return struct {
        const Input = TrainingExample(Net.Input, f32);
        const Output = f32;
        const Param = Net.Param;

        fn initializeParams(param: *Param, rng: *std.rand.Random) void {
            Net.initializeParams(param, rng);
        }

        fn run(input: *Input, param: *Param, output: *Output) void {
            var predicted = [1]f32{0};
            Net.run(&input.input, param, &predicted);
            var loss = (predicted[0] - input.target) * (predicted[0] - input.target);
            output.* += loss;
        }
        fn reverse(
            input: *Input,
            param: *Param,
            delta: *Output,
            backprop: *Input, // add
            gradient: *Param, // add
        ) void {
            // 'delta' indicates how much being wrong counts.
            // the amount we pass back into the previous layer is therefore based
            // on how far off the current estimate is.

            // So we first need to run forward to obtain a prediction:
            var predicted = [1]f32{0};
            Net.run(&input.input, param, &predicted);

            // We have: L = (pred - target)^2
            // and we know dE / dL
            // we want to find dpred / dL

            // dE/dA = dE/dL dL/dA
            // so take d/dpred of both sides:
            // dL/dpred = 2(pred-target)

            // so dE/dpred = dE/dL 2 (pred - target).

            var adjustedDelta = 2 * delta.* * (predicted[0] - input.target);
            var discardInputBackprop = zeroed(Net.Input);
            Net.reverse(
                &input.input,
                param,
                &adjustedDelta,
                &discardInputBackprop,
                gradient,
            );
        }
    };
}

pub fn main() !void {
    const stdout = std.io.getStdOut().outStream();

    const NetPredict = Fan(1, Lin(1)); // Fan of 1 converts to [1]f32
    const Net = LossSum(LossL2(NetPredict));

    var rng = std.rand.Pcg.init(0);

    var params = zeroed(Net.Param);
    Net.initializeParams(&params, &rng.random);

    std.debug.warn("param :: {}\n", .{params[0]});

    var training = [_]TrainingExample([1]f32, f32){
        .{ .input = [1]f32{0}, .target = 0 },
        .{ .input = [1]f32{1}, .target = 2 },
        .{ .input = [1]f32{2}, .target = 4 },
        .{ .input = [1]f32{-1}, .target = -2 },
        .{ .input = [1]f32{-2}, .target = -4 },
    };

    var output: f32 = 0;
    var runtime_zero: usize = 0;
    var training_slice: []TrainingExample([1]f32, f32) = training[runtime_zero..training.len];
    Net.run(&training_slice, &params, &output);
    try stdout.print("loss :: {d:3.2}\n", .{output});

    var gradient = zeroed(Net.Param);
    var backpropDiscard = zeroed(Net.Input);
    Net.reverse(&training_slice, &params, &output, &backpropDiscard, &gradient);

    try stdout.print("gradient :: weight : {d:3.2} ; bias : {d:3.2}\n", .{ gradient[0].weights[0], gradient[0].bias });
}
