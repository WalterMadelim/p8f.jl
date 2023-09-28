j = 1260
d = 9
for i in j:j+d
    for y in 2020:2023
        local url = "https://pubsonline.informs.org/doi/pdf/10.1287/ijoc." * "$y" * "." * "$i" * "?download=true"
        local command = `cmd /c start $url`
        run(command)
    end
end
