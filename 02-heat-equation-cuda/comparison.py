import subprocess, re

# comparison over 3 different grid sizes
configs_t3 = [
    (250,  16, 16),
    (500,  16, 32),
    (1000, 16, 63),
]
iters = 1000

print("| Versão | Tamanho ($n$) | Tempo CPU (ms) | GPU Kernel (ms) | GPU Total (ms) | Speedup Total | Speedup Kernel | Validação |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for n, t, b in configs_t3:
    for label, binary in [("Tarefa 2 (Sem corpo)", "./heat"), ("Tarefa 3 (Corpo 37°C)", "./heat_tarefa3")]:
        res = subprocess.run([binary, str(n), str(iters), str(t), str(b)], capture_output=True, text=True)
        out = res.stdout
        val = re.search(r"Validation:\s+(\w+)", out).group(1)
        cpu = re.search(r"CPU Time:\s+([\d.]+)", out).group(1)
        kern = re.search(r"GPU Kernel Time:\s+([\d.]+)", out).group(1)
        tot = re.search(r"GPU Total Time:\s+([\d.]+)", out).group(1)
        sp_tot = re.search(r"Speedup \(Total\):\s+([\d.]+x)", out).group(1)
        sp_kern = re.search(r"Speedup \(Kernel\):\s+([\d.]+x)", out).group(1)
        print(f"| **{label}** | {n} | {cpu} | {kern} | {tot} | **{sp_tot}** | **{sp_kern}** | {val} |")