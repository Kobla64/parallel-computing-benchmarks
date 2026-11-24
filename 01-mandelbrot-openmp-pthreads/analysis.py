import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Daten einlesen
df = pd.read_csv("measurements.csv")

# Layout vorbereiten (2x2 Raster)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# ----------------------------------------------------
# Diagramm 1: Laufzeit vs. Bildgröße (Full Picture)
# ----------------------------------------------------
ax1 = axes[0, 0]
df_full = df[df['region'] == 'full']

seq_no_io = df_full[(df_full['version'] == 'seq') & (df_full['io'] == 1)]
ax1.errorbar(seq_no_io['image_size'], seq_no_io['mean_time_sec'], yerr=seq_no_io['std_dev_sec'],
             label='Sequencial (sem I/O)', marker='o', capsize=3)

pth_4 = df_full[(df_full['version'] == 'pth') & (df_full['threads'] == 4)]
ax1.errorbar(pth_4['image_size'], pth_4['mean_time_sec'], yerr=pth_4['std_dev_sec'],
             label='Pthreads (4 threads)', marker='s', capsize=3)

omp_4 = df_full[(df_full['version'] == 'omp') & (df_full['threads'] == 4)]
ax1.errorbar(omp_4['image_size'], omp_4['mean_time_sec'], yerr=omp_4['std_dev_sec'],
             label='OpenMP (4 threads)', marker='^', capsize=3)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Tamanho da Imagem (N x N pixels)')
ax1.set_ylabel('Tempo Médio (s)')
ax1.set_title('1. Tempo de Execução vs. Tamanho da Imagem (Full)')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ----------------------------------------------------
# Diagramm 2: Speedup vs. Thread-Anzahl (4096 x 4096)
# ----------------------------------------------------
ax2 = axes[0, 1]
size_target = 4096
t_seq_full = df[(df['region'] == 'full') & (df['version'] == 'seq') & 
                (df['io'] == 1) & (df['image_size'] == size_target)]['mean_time_sec'].values[0]

pth_4096 = df[(df['region'] == 'full') & (df['version'] == 'pth') & 
               (df['image_size'] == size_target)].sort_values('threads')
omp_4096 = df[(df['region'] == 'full') & (df['version'] == 'omp') & 
               (df['image_size'] == size_target)].sort_values('threads')

speedup_pth = t_seq_full / pth_4096['mean_time_sec']
speedup_omp = t_seq_full / omp_4096['mean_time_sec']
threads = pth_4096['threads']

ax2.plot(threads, speedup_pth, marker='s', label='Pthreads (4096²)')
ax2.plot(threads, speedup_omp, marker='^', label='OpenMP (4096²)')
ax2.plot(threads, threads, 'k--', alpha=0.5, label='Speedup Ideal Linear')

ax2.set_xlabel('Número de Threads')
ax2.set_ylabel(r'Speedup ($S_p = T_{seq} / T_p$)')
ax2.set_title('2. Speedup vs. Número de Threads (4096²)')
ax2.set_xticks([1, 2, 4, 8, 16, 32])
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend()

# ----------------------------------------------------
# Diagramm 3: Vergleich der 4 Regionen (Sequenziell)
# ----------------------------------------------------
ax3 = axes[1, 0]
regions = ['full', 'seahorse', 'elephant', 'triple_spiral']
region_labels = ['Full', 'Seahorse', 'Elephant', 'Triple Spiral']
sizes_to_plot = [512, 1024, 2048, 4096]

x = np.arange(len(region_labels))
width = 0.2
for idx, s in enumerate(sizes_to_plot):
    times = [df[(df['region'] == r) & (df['version'] == 'seq') & 
                (df['io'] == 1) & (df['image_size'] == s)]['mean_time_sec'].values[0] for r in regions]
    ax3.bar(x + idx * width - 1.5 * width, times, width, label=f'{s}x{s}')

ax3.set_xticks(x)
ax3.set_xticklabels(region_labels)
ax3.set_ylabel('Tempo Médio (s)')
ax3.set_title('3. Comparação de Tempo por Região (Sequencial)')
ax3.grid(True, axis='y', ls="--", alpha=0.5)
ax3.legend()

# ----------------------------------------------------
# Diagramm 4: Einfluss von I/O (Sequenziell: io=0 vs io=1)
# ----------------------------------------------------
ax4 = axes[1, 1]
seq_with_io = df_full[(df_full['version'] == 'seq') & (df_full['io'] == 0)].sort_values('image_size')
seq_without_io = df_full[(df_full['version'] == 'seq') & (df_full['io'] == 1)].sort_values('image_size')

ax4.plot(seq_with_io['image_size'], seq_with_io['mean_time_sec'], marker='o', label='Com I/O (io=0)')
ax4.plot(seq_without_io['image_size'], seq_without_io['mean_time_sec'], marker='x', label='Sem I/O (io=1)')

ax4.set_xscale('log', base=2)
ax4.set_yscale('log')
ax4.set_xlabel('Tamanho da Imagem (N x N pixels)')
ax4.set_ylabel('Tempo Médio (s)')
ax4.set_title('4. Impacto de I/O na Versão Sequencial (Full)')
ax4.grid(True, which="both", ls="--", alpha=0.5)
ax4.legend()

plt.tight_layout()
plt.savefig('mandelbrot_benchmarks.png', dpi=300)
print("Grafik mandelbrot_benchmarks.png wurde erfolgreich erzeugt!")