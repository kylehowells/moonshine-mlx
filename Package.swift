// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "moonshine-mlx",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MoonshineMLX", targets: ["MoonshineMLX"]),
        .executable(name: "moonshine-mlx", targets: ["moonshine-mlx-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "MoonshineMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "moonshine-mlx-cli",
            dependencies: [
                "MoonshineMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "SwiftTrace",
            dependencies: [
                "MoonshineMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftTrace"
        ),
    ]
)
