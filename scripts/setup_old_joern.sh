#!/usr/bin/env bash
# setup_old_joern.sh — ATTEMPT to install the 2014 Yamaguchi joern stack VulPCL's graph_generation.py
# needs (joern + joern-tools + python-joern + Neo4j 2.1.8 + Gremlin + graphviz).
#
# !!! HIGH RISK / ARCHAEOLOGY !!! Python 2 is EOL, Neo4j 2.1.8 + gremlin-plugin are ancient, the joern
# ant build targets old Java. Any step may fail. This is the documented path, best-effort. If it
# dead-ends, pivot to option B (modern-joern re-derive). Run on a fresh pod; iterate per [TRY].
set -uo pipefail   # NOT -e: we want to keep going + see which step breaks
OPT=/opt; mkdir -p "$OPT"

echo "=== [1] system deps: JDK8, ant, graphviz, python2 ==="
apt-get update -q
apt-get install -y -q openjdk-8-jdk ant graphviz git wget unzip curl python2 || \
  apt-get install -y -q openjdk-8-jdk ant graphviz git wget unzip curl python2.7
export JAVA_HOME=$(ls -d /usr/lib/jvm/java-8-openjdk* 2>/dev/null | head -1)
echo "  JAVA_HOME=$JAVA_HOME"

echo "=== [2] pip2 (python2 EOL bootstrap) ==="
curl -sS https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /tmp/get-pip.py
python2 /tmp/get-pip.py || echo "  [TRY] pip2 bootstrap failed — python2 may be unavailable on this image"

echo "=== [3] python-joern + joern-tools + py2neo (Python 2) ==="
# These provide joern.all.JoernSteps + the joern-lookup / joern-plot-* shell tools.
python2 -m pip install --no-cache-dir "py2neo==2.0" python-joern joern-tools pexpect \
  || echo "  [TRY] py2 packages failed — try installing from the octopus-platform github clones"

echo "=== [4] build joern (octopus-platform, ant + JDK8) ==="
[[ -d "$OPT/joern" ]] || git clone --depth 1 https://github.com/octopus-platform/joern "$OPT/joern"
( cd "$OPT/joern" && JAVA_HOME="$JAVA_HOME" ant ) || echo "  [TRY] ant build failed — check JDK8 + ant versions"
export PATH="$PATH:$OPT/joern/bin:$HOME/.local/bin"

echo "=== [5] Neo4j 2.1.8 + gremlin-plugin ==="
if [[ ! -d "$OPT/neo4j-community-2.1.8" ]]; then
  wget -q "https://dist.neo4j.org/neo4j-community-2.1.8-unix.tar.gz" -O /tmp/neo4j.tar.gz \
    || wget -q "https://neo4j.com/artifact.php?name=neo4j-community-2.1.8-unix.tar.gz" -O /tmp/neo4j.tar.gz \
    || echo "  [TRY] Neo4j 2.1.8 download failed — find the legacy tarball mirror"
  tar xzf /tmp/neo4j.tar.gz -C "$OPT" 2>/dev/null || true
fi
# gremlin-plugin (server plugin for Neo4j 2.x) — joern's runGremlinQuery needs it.
echo "  [TRY] install neo4j-contrib/gremlin-plugin jar into $OPT/neo4j-community-2.1.8/plugins/ + enable in conf"

echo "=== [6] verify the tools VulPCL calls are on PATH ==="
for t in joern joern-lookup joern-plot-proggraph joern-plot-ast dot; do
  command -v "$t" >/dev/null && echo "  OK  $t -> $(command -v $t)" || echo "  MISSING  $t  [TRY]"
done

cat <<'EOF'

=== NEXT (manual, per VulPCL graph_generation.py) ===
  1) parse our code into a joern index:
       java -jar /opt/joern/bin/joern.jar  src/VulPCL/vul_categorization/vul_data/code/megavul
     -> produces .joernIndex
  2) load it into Neo4j 2.1.8 + start the server:
       cp -r .joernIndex /opt/neo4j-community-2.1.8/data/graph.db
       /opt/neo4j-community-2.1.8/bin/neo4j start
  3) confirm:  python2 -c "from joern.all import JoernSteps; j=JoernSteps(); j.connectToDatabase(); print('ok')"
  4) then re-run scripts/run_vulpcl_megavul_oldjoern.sh  (passes the [WALL], runs graph_generation -> svg)

If any [TRY]/MISSING above won't resolve -> this 2014 stack won't build here; use option B (modern joern).
EOF
