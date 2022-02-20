from grpc.tools import protoc


protoc.main(
    (
        '',
        '-I.',
        '--python_out=.',
        '--grpc_python_out=.',
        './xarm_shuidi_grpc.proto',
    )
)