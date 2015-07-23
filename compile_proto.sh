PROTO_SRC_DIR=src/proto
DST_DIR=build
PROTO_BUILD_DIR=$DST_DIR/$PROTO_SRC_DIR
PY_PROTO_BUILD_DIR=python/gps/proto

mkdir -p "$PROTO_BUILD_DIR"
mkdir -p "$PY_PROTO_BUILD_DIR"
touch $PY_PROTO_BUILD_DIR/__init__.py

protoc -I=$PROTO_SRC_DIR --cpp_out=$DST_DIR $PROTO_SRC_DIR/gps.proto
protoc -I=$PROTO_SRC_DIR --python_out=$PY_PROTO_BUILD_DIR $PROTO_SRC_DIR/gps.proto
