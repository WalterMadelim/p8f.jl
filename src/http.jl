addr = "https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=9655379&punumber=5165391"

addr = "https://pubsonline.informs.org/toc/ijoc/current"
resp = HTTP.get(addr)



addr = "https://fanyi.baidu.com/#en/zh/scrape"
addr = "https://pubsonline.informs.org/"

addr = "https://www.siam.org/"
headers=Dict("User-Agent" => "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.43")
r = HTTP.request("POST", addr; headers)

println(r.status)

println(String(r.body))

url = "https://pubsonline.informs.org/"
response = HTTP.get(url; headers)

open("issue4.txt","w") do io
    println(io,resp.status)
    println(io,"%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println(io,String(resp.body))
end


[1290,1291,1274,1292,1286,1282,1287,1289,1294,1295]




