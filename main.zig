const std = @import("std");
const net = @import("./net.zig");

pub fn main() !void {
    const stdout = std.io.getStdOut().outStream();

    const NetPredict = net.Fan(1, net.Lin(1)); // Fan of 1 converts to [1]f32
    const Net = net.LossSum(net.LossL2(NetPredict));

    var rng = std.rand.Pcg.init(0);

    var training = [_]net.TrainingExample([1]f32, f32){
        .{ .input = [1]f32{0}, .target = 0 },
        .{ .input = [1]f32{1}, .target = 2 },
        .{ .input = [1]f32{2}, .target = 4 },
        .{ .input = [1]f32{-1}, .target = -2 },
        .{ .input = [1]f32{-2}, .target = -4 },
    };

    var params = net.zeroed(Net.Param);
    Net.initializeParams(&params, &rng.random);

    var iter = net.RangeTo(1000).new();
    while (iter.next()) |i| {
        var output: f32 = 0;
        Net.run(&training, params, &output);
        try stdout.print("loss :: {d:3.2}\n", .{output});

        var gradient = net.zeroed(Net.Param);
        var backpropDiscard = net.zeroed(Net.Input);
        Net.reverse(&training, params, 1, &backpropDiscard, &gradient);

        Net.updateGradient(&params, -0.01, &gradient);
    }
}
