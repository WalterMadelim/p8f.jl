j = 1270 # starting vol num
d = 9 # try 10 files at one loop
for i in j:j+d
    for y in 2020:2023 # year range
        local url = "https://pubsonline.informs.org/doi/pdf/10.1287/ijoc." * "$y" * "." * "$i" * "?download=true"
        command = `cmd /c start $url`
        run(command)
    end
end
