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

pub fn main() !void {
    const stdout = std.io.getStdOut().outStream();

    const Net = Relu(3, 2);
    var params: Net.Param = zeroed(Net.Param);

    var rng = std.rand.Pcg.init(0);
    Net.initializeParams(&params, &rng.random);

    var input: Net.Input = .{
        100,
        110,
        120,
    };

    var output = [2]f32{ 0, 0 };
    Net.run(&input, &params, &output);
    try stdout.print("out :: {d:3.2}\n", .{output});

    var delta: Net.Output = [2]f32{ 1, 1 };
    var backprop: Net.Input = .{ 0, 0, 0 };
    var gradient: Net.Param = zeroed(Net.Param);

    Net.reverse(&input, &params, &delta, &backprop, &gradient);
    try stdout.print("out :: {d:3.2} {d:3.2}\n", .{ output[0], output[1] });
}
