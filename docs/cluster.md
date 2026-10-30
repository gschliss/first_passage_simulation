# Running a larger sweep on a cluster

How to clone this project onto a SLURM cluster, reproduce the D = 0.5 matrix
(`docs/report_d05.md`), and then grow it. Read `docs/handoff.md` first for the model itself;
this file is only about getting it to run somewhere bigger.

**On Sherlock (Stanford), read §10 first** — it gives the concrete paths, modules and
partition for that cluster, and the rest of this file then reads as background.

**The shape of the problem.** One *cell* is one `dimension_experiment.py` invocation: a
switching walker and its free-only reference, over the same shared receptor layouts, at one
point in parameter space. Cells are completely independent, so a matrix of them is a plain
SLURM array. `scripts/sweep.py` enumerates the cells; `slurm/submit_matrix.sh` runs one per
array task.

Everything is resumable. Each (sweep point, layout) chunk is a separate Parquet file and
`run_task` skips chunks already on disk, so a failed or timed-out task is fixed by submitting
the same array again.

---

## 1. Pick where it goes, and clone

Clusters usually pair a small quota'd home with a large scratch or project filesystem. The
code is tiny; the output is not (~1.5 MB per cell), so **clone somewhere with room** —
`/projects/$USER`, `/work/$USER`, whatever yours provides — rather than home:

```bash
cd /projects/$USER            # or /work/$USER, /scratch/$USER ...
git clone https://github.com/gschliss/first_passage_simulation.git
cd first_passage_simulation
```

The repo is **public** (made so on 2026-10-29), so that clone needs no credentials at all and
nothing below is required to get started. Read on only if you want to **push** from the
cluster, or if the repo is made private again.

### If the machine is already signed in to GitHub

Find out which mechanism it has, and use the matching URL:

```bash
ssh -T git@github.com                  # SSH key  -> "Hi gschliss! You've successfully authenticated"
gh auth status                         # gh CLI   -> then: gh repo clone gschliss/first_passage_simulation
git config --get credential.helper     # stored token -> the https:// URL works directly
```

### If it is not signed in

**Preferred: a deploy key.** It is scoped to this one repository and revocable on its own, so
a shared cluster never holds credentials for the rest of your account. Leave it read-only if
you only ever pull there — results come home by `rsync`, not through git — and tick "Allow
write access" only if you intend to commit from the cluster.

On the cluster:

```bash
ssh-keygen -t ed25519 -C "fpsim deploy key (cluster)" -f ~/.ssh/id_ed25519_fpsim -N ""
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519_fpsim
cat ~/.ssh/id_ed25519_fpsim.pub
```

`-N ""` leaves the key without a passphrase. That is the usual trade for an unattended
machine and is acceptable here *because* the key is read-only and single-repo; nothing in the
job scripts does network git, so a passphrase would only ever be prompting you interactively.

Paste the printed public key at
**github.com/gschliss/first_passage_simulation/settings/keys** → *Add deploy key*, ticking
"Allow write access" only if you need to push. Then tell SSH to use it, since it is not a
default key name:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/id_ed25519_fpsim
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -T git@github.com
```

A deploy key authenticates to the *repository*, not to you, so the greeting reads **"Hi
gschliss/first_passage_simulation!"** rather than your username. That is success, not a
misconfiguration. Now clone with the SSH URL as above.

**Alternative: `gh auth login`.** Pick *Login with a web browser*; it prints a one-time code
you enter on your laptop, so it works over a plain SSH session with no browser on the
cluster. Follow it with `gh auth setup-git` to make `https://` clones work too. Simpler than
a deploy key, but it signs the cluster in to your whole account.

**Alternative: a fine-grained token over HTTPS.** Create one at
*github.com/settings/personal-access-tokens*, scoped to this repository with *Contents:
Read-only*, and give it as the password when git prompts. Do **not** run
`git config --global credential.helper store` on a cluster — it writes the token in plaintext
to `~/.git-credentials` on a shared filesystem. Use `credential.helper 'cache --timeout=3600'`
if you want to avoid retyping it; that keeps it in memory only.

### If the cluster has no outbound access to GitHub at all

Some sites firewall even the login nodes. Carry the repository over as a bundle — a single
file holding the complete history:

```bash
# on your laptop, in the repo
git bundle create /tmp/fpsim.bundle --all
scp /tmp/fpsim.bundle cluster:/projects/$USER/
```

```bash
# on the cluster
cd /projects/$USER
git clone -b main fpsim.bundle first_passage_simulation
cd first_passage_simulation
git remote set-url origin git@github.com:gschliss/first_passage_simulation.git
```

The bundle is ~1.3 MB for this repo and clones to a normal working checkout. The last line
matters: without it `origin` points at the bundle file, which no longer updates. To refresh
later, make a new bundle and `git pull /path/to/new.bundle main`.

### Keeping the output off a purged or quota'd filesystem

If scratch is purged on a schedule, keep the clone on a persistent filesystem and point only
the output at scratch:

```bash
mkdir -p /scratch/$USER/fpsim_results
rm -rf results && ln -s /scratch/$USER/fpsim_results results
```

`git status` will then show `results/.gitkeep` as deleted and `results` as an untracked
symlink. That is harmless — nothing in the code cares — but don't commit it.

## 2. Build the environment

Either conda:

```bash
conda env create -f environment.yml
conda activate fpsim
```

or a venv against a Python ≥ 3.9 module:

```bash
module avail python                # see what this cluster actually has
module load python/3.11
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**No compiler or `module load gcc` is needed.** On x86_64 and aarch64 Linux every dependency
(numpy, scipy, numba/llvmlite, pyarrow, matplotlib, pandas) installs from a binary wheel. If
pip starts building something from source you are on an unusual architecture or an old pip —
try `pip install --upgrade pip` first. **Or you are on an old glibc** (CentOS/RHEL 7, glibc 2.17 —
Sherlock is one): the newest numpy ships only `manylinux_2_28` wheels, so pip falls back to a
source build that fails. `pip install --prefer-binary -e ".[dev]"` picks the newest release
that has a wheel for your glibc instead (numpy 2.2.6 on Sherlock, 2026-10). Check with
`ldd --version`.

```bash
mkdir -p slurm/logs                # SLURM will not create this, and jobs fail without it
```

Edit the two placeholder blocks in `slurm/submit_matrix.sh`: the partition/account `#SBATCH`
lines, and the environment activation line.

## 3. Check the install before spending an allocation

Three steps, cheapest first.

```bash
python scripts/check_env.py
```

Reports the Python version, every dependency and its version, whether `fpsim` imports, how
many cores the process may *really* use (cgroups on a cluster often cut this well below
`nproc`), whether Numba compiles and runs a parallel kernel, and whether the results
filesystem supports the hard link `jobs.write_once` needs. Exits non-zero on any FAIL. Run it
again inside a job — a login node and a compute node can differ on all of those.

```bash
NUMBA_NUM_THREADS=8 python -m pytest -q        # 73 tests, ~15 s after Numba compiles
```

Then reproduce one published number end to end. Base task 1 is the ring-2 reference cell:

```bash
python scripts/sweep.py --plan matrix_d05 --stage base --array-task 1
```

About 10 minutes on 8 cores. It should print a `paired_ratios` line with **t20_switch ≈ 211.5
s**, **t20_ref ≈ 297.5 s**, captured **1.000** and **0.758** — the numbers in the report's
headline. If those match, the cluster build agrees with the machine the report was written on.

Results do not depend on the thread count: each molecule draws from its own seeded RNG, which
was checked bit-for-bit across 6 and 8 threads (L89). `--cpus-per-task` is therefore free to
differ from the 8 used locally.

## 4. Look at the plan

```bash
python scripts/sweep.py --plan matrix_d05 --list
```

87 cells in 19 *reuse groups*. A group is a set of cells that share one free-only reference:
the reference never binds, so it does not depend on the free fraction or the dwell value (L95,
verified bit-identical). It does depend on the ring, receptor count, `--finds`, `--d-ref`,
`--n`, `--layouts`, `t_max` and the *number* of dwells.

That matters because **the reference is ~97% of the compute** (L95: 9.8 min at N = 1 and 19.9
min at N = 5 per 400 molecules, against 0.3 min for the switching arm). So there are two ways
to run a matrix:

| | arrays | core-hours | wall time |
|---|---|---|---|
| `--stage all` | one, no dependency | ~5× | one slowest cell |
| `--stage base` then `--stage rest` | two, second depends on first | ~1× | two slowest cells |

**Use `--stage all` unless your allocation is tight.** On a cluster the extra core-hours buy
you a single dependency-free array, and the whole matrix lands in the time of its slowest cell
(~25 min for this plan). The two-stage path exists for when core-hours are the scarce thing.

## 5. Submit

Simple path:

```bash
N=$(python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks)
sbatch --array=0-$((N-1))%40 slurm/submit_matrix.sh matrix_d05 all
```

The `%40` caps concurrent tasks; drop or raise it to suit the queue.

Two-stage path, with SLURM enforcing the ordering:

```bash
NB=$(python scripts/sweep.py --plan matrix_d05 --stage base --print-n-tasks)
NR=$(python scripts/sweep.py --plan matrix_d05 --stage rest --print-n-tasks)
JOB=$(sbatch --parsable --array=0-$((NB-1)) slurm/submit_matrix.sh matrix_d05 base)
sbatch --dependency=afterok:$JOB --array=0-$((NR-1)) slurm/submit_matrix.sh matrix_d05 rest
```

`--stage rest` resolves each cell's reference directory by asking the driver
(`--print-out-dir`), so the two stages agree on naming without duplicating the rules.

### Resubmitting

Tasks skip finished chunks, so after failures just submit the same array again. To find which
cells are incomplete:

```bash
python scripts/sweep.py --plan matrix_d05 --stage all --list | tail -n +2 | wc -l
ls -d results/dimension/*_d05/paired_ratios.csv | wc -l     # how many finished
```

## 6. Collect the results

```bash
bash scripts/make_figures.sh        # writes docs/figures/, prints a cell count per figure
```

`results/` is gitignored and never leaves the cluster on its own. Either run the figure script
there and copy back `docs/figures/`, or bring the raw output home:

```bash
rsync -av --info=progress2 cluster:first_passage_simulation/results/dimension/ results/dimension/
```

Output is small: ~1.5 MB per cell (40 Parquet chunks plus the per-run figure), 154 MB for the
whole 87-cell matrix.

## 7. Growing the sweep

Edit `PLANS` at the top of `scripts/sweep.py`. A plan is `common` (flags every cell gets) plus
`blocks`, and each block is a cross product of axis values. Cells that come out identical
across blocks are deduplicated, so overlapping blocks are fine and intentional.

To widen an existing axis, add values:

```python
FREES = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
```

To add a region, add a block — e.g. the dwell × capture-efficiency interaction the report flags
as unexplored:

```python
dict(free_fraction=[0.05], finds=[1, 2, 5, 10], target_ring=[2],
     receptors_per_cell=[1000], dwells=["0.1", "1", "10", "100"]),
```

Then re-check the size and the grouping before submitting:

```bash
python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks
python scripts/sweep.py --plan matrix_d05 --list | head -30
```

Two things to keep in mind when you enlarge it:

- **Task indices are positional.** Adding values renumbers cells. That is harmless for a fresh
  submission (finished cells are skipped anyway) but means an array id is not a durable name
  for a cell.
- **The figure globs in `scripts/make_figures.sh` are tied to directory naming.** A new axis
  changes directory names and can silently widen a glob — exactly the bug that put `finds2` and
  `finds10` runs into the N = 1 figures in L96. Duplicate cells are now a hard error in
  `plot_grid.py` and `plot_line.py`, so a bad glob fails loudly, but you still have to add the
  globs for any new figure.

### One cell that is too big for one node

A cell runs its chunks in one process, so its ceiling is a single node. If you raise `--n` far
enough that one cell no longer fits in a time limit, split it with `--shard K/M`:

```bash
for k in 0 1 2 3; do
  python scripts/dimension_experiment.py <cell flags> --shard $k/4 &    # or 4 array tasks
done; wait
python scripts/dimension_experiment.py <cell flags> --replot            # summarise and plot
```

Shard `K/M` runs every M-th chunk starting at K. It is a scheduling flag only — it never
reaches the config, so it cannot change a hash or a result (checked bit-for-bit against an
unsharded run). The `--replot` pass at the end reads whatever is on disk and writes
`summary.csv`, `paired_ratios.csv` and the per-run figure.

Pass it through `sweep.py` with `--extra`:

```bash
python scripts/sweep.py --stage all --array-task 7 --extra --shard 0/4
```

## 8. Cluster-specific gotchas

The model gotchas are in `docs/handoff.md` §5 and still apply. These are the ones that only
show up on a cluster:

- **Numba's compile cache on a shared filesystem.** Concurrent tasks writing one cache
  directory corrupt it. `submit_matrix.sh` gives each task its own `NUMBA_CACHE_DIR` on
  node-local disk (`$L_SCRATCH` if the site sets it, else `$TMPDIR`) and deletes it on exit.
  Keep that if you write your own submit script.
- **`results/` must be on a filesystem that supports `os.link`.** `jobs.write_once` uses a
  hard link to make `config.json` creation atomic between concurrent tasks. Scratch and home
  filesystems are fine; some FUSE mounts are not.
- **One code version per results directory.** Every chunk records its git commit, and
  `jobs.merge` reports `chunks come from several code versions` if they disagree. Don't pull
  mid-sweep; finish, then pull.
- **Memory is small and flat:** peak RSS ~190 MB even at ring 4 with 10 000 receptors per
  cell. The 4 GB in the submit script is deliberate headroom, not a measured requirement. Ask
  for fewer cores per task and more tasks if the queue rewards it.
- **Wall-clock per cell is dominated by the reference arm and by `--finds`**, not by the free
  fraction. The 0.1%-free corner is cheap (L95), contrary to what you might expect.
- **`validate_3d_capture.py` step 4 is broken**: it hardcodes a run directory deleted in L91.
  Use `--skip-lattice` until it is repointed.

## 9. The older per-config array path

`slurm/submit_sweep.sh` and `scripts/run.py` predate the paired design. They run one *chunk*
per array task from a config file's own `sweeps:` block, writing to `results/runs/<name>/`, and
have no switching-vs-reference pairing, no shared layouts and no t_20. They are still the right
tool for a plain single-walker sweep defined entirely in YAML. Everything in the report came
through the `sweep.py` path described above.

## 10. Sherlock (Stanford)

Sherlock is SLURM, so everything above applies. These are the parts that differ, checked
against Stanford Research Computing's own docs (October 2026).

**Cloning needs nothing**, since the repo is public: the `https://` URL in §1 works as-is.
`gh` is not installed on Sherlock, and Sherlock has no SSH key for GitHub out of the box, so
`gh auth login` and `git@github.com:` URLs both fail there (`Permission denied (publickey)`).
Neither matters unless you want to push from the cluster, in which case use the deploy key
from §1 — it needs only `ssh-keygen` and `git`, which are always present.

If you do set up SSH and `ssh -T git@github.com` hangs or is refused, outbound port 22 is
blocked. GitHub serves
SSH on 443 as well, which gets through:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/id_ed25519_fpsim
  IdentitiesOnly yes
EOF
```

**Where things go.** Sherlock gives you several filesystems with very different quotas:

| | quota | notes | use it for |
|---|---|---|---|
| `$HOME` | 15 GB | NFS, not for parallel I/O | nothing here, it is too small to be worth the risk |
| `$GROUP_HOME` | 1 TB | shared with your PI group | **the clone and the venv** |
| `$SCRATCH` | 100 TB | files expire after a period without modification | **`results/`** |
| `$OAK` | purchased | long-term, if your lab has it; **"not designed for bursty or parallel job I/O"** | the clone, and finished results copied back off `$SCRATCH` |
| `$L_SCRATCH` | node-local SSD | deleted when the job ends | Numba's compile cache (the submit script already uses it) |

The code can sit on `$GROUP_HOME` or, if your lab has it, on `$OAK` — it is small and read
once per task, so either is fine. What matters is that **`results/` goes on `$SCRATCH`**:
dozens of array tasks writing Parquet chunks at once is precisely the parallel I/O that
`$OAK` and the NFS homes are not built for.

```bash
cd $OAK/<you>/...            # or $GROUP_HOME
git clone https://github.com/gschliss/first_passage_simulation.git
cd first_passage_simulation
mkdir -p $SCRATCH/fpsim_results
rm -rf results && ln -s $SCRATCH/fpsim_results results
mkdir -p slurm/logs
```

Mind the `$SCRATCH` purge policy: it keys off last *modification*, and finished Parquet
chunks are never rewritten, so a matrix you leave alone will eventually age out. When a sweep
finishes, copy it somewhere permanent — which is what `$OAK` is for:

```bash
rsync -a $SCRATCH/fpsim_results/ $OAK/<you>/fpsim_results/
```

Put the **virtual environment on `$GROUP_HOME`**, not on `$OAK` and not inside a repo that
lives there — Stanford's docs ask for venvs on `$HOME` or `$GROUP_HOME`. An editable install
works fine with the two split apart:

```bash
python3 -m venv $GROUP_HOME/fpsim-venv
source $GROUP_HOME/fpsim-venv/bin/activate
pip install --prefer-binary -e ".[dev]"   # run from the checkout; --prefer-binary: see §2
```

**Build the environment in an `sh_dev` session, not on a login node.** Stanford asks you to
keep login nodes free, and says outright that building environments there is too heavy:

```bash
sh_dev -c 4 -m 8GB          # a compute node: 1 h, which is plenty
ml avail python             # see the stack as it is today
ml python/3.12.1
python3 -m venv $GROUP_HOME/fpsim-venv   # python3: plain `python` is an old system interpreter
source $GROUP_HOME/fpsim-venv/bin/activate
pip install --upgrade pip
pip install --prefer-binary -e ".[dev]"   # glibc 2.17 here: see §2
export NUMBA_NUM_THREADS=4  # match -c above, or check_env.py warns about oversubscription
python scripts/check_env.py
python -m pytest -q
```

Sherlock's system git is 1.8, too old for `git -C`; `fpsim.io.git_commit` and `check_env.py`
use `cwd=` instead, so chunk metadata gets the commit without `ml git`.

Once the venv is activated, plain `python` is the right one, which is why the commands
elsewhere in this file work unchanged from there.

**Submitting.** `sh_part` lists the partitions you can actually use. The `normal` partition
defaults to a 2-hour limit and allows 2 days, so the 6 hours in `submit_matrix.sh` is fine
but must be stated explicitly — it is above the default. If your lab owns nodes, `owners`
allows 7 days but your jobs there can be preempted, which is safe for this work since every
task resumes from the chunks already on disk.

Add the partition to `slurm/submit_matrix.sh` (uncomment and edit the `#SBATCH --partition`
line), set the activation line to:

```bash
ml python/3.12.1
source $GROUP_HOME/fpsim-venv/bin/activate
```

and submit as in §5:

```bash
N=$(python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks)
sbatch --array=0-$((N-1))%40 slurm/submit_matrix.sh matrix_d05 all
```

**Run `check_env.py` again inside a job**, not just in `sh_dev`. It reports the cores the
process may really use, and a compute node under a cgroup can differ from an interactive
session.
