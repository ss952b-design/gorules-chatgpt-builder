import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.useWasmCache = true;
env.logLevel = 40;

var chatHistory=[];
var lastGraph=null;
var generator=null;
var activeModel=null;
var generating=false;

function $(id){return document.getElementById(id);}
var SYSTEM=[
"You are a GoRules BRMS architect. Build production-oriented GoRules decision graph JSON from the user's requirement.",
"Return exactly one JSON object with nodes, edges, validationNotes. No markdown.",
"Allowed node types: inputNode, expressionNode, decisionTableNode, functionNode.",
"Use functionNode ONLY for external API/network calls; never business logic.",
"Use expressionNode for calculations/transformations and decisionTableNode for thresholds, categories and pass/refer/reject rules.",
"ZEN expressions must not contain JavaScript-only syntax: spread, optional chaining, nullish coalescing, arrow functions, JS Array method chains, or arbitrary JavaScript.",
"Prefer conservative GoRules ZEN functions when supported: map, filter, flatten, sum, avg, min, max, len, number, string, round, floor, ceil and supported date functions.",
"Keep IDs unique, every edge source and target must reference existing node IDs, and expressions must avoid undefined variables.",
"For scorecards/BREs: when a parameter has a present extracted value, its score must be numeric, its decision must be PASS/REFER/REJECT, and its score must come from an explicit decisionTableNode rule. A parameter result should follow {parameter, extractedValue, score, maxScore, decision} and the final result should expose totalScore, totalMaxScore, scorePercent, counts, finalDecision and riskCategory. N/A is allowed only when the underlying value is missing, null, blank, or genuinely unavailable. A score of 0 is valid but must never be paired with decision N/A. Include parameter-level score, maxScore, decision, plus aggregate totalScore, totalMaxScore, scorePercent, counts for PASS/REFER/REJECT/N/A, finalDecision and riskCategory when the requirement asks for scoring. Use FIRST_MATCH decision tables and add a final catch-all row where appropriate. For vehicle credit scorecards, do not silently drop scoring outputs after extracting values; wire decision-table outputs through the graph to the final result."
].join("\n");

function add(role,text){
  var d=document.createElement("div");
  d.className="msg "+role;
  d.textContent=text;
  $("messages").appendChild(d);
  $("messages").scrollTop=$("messages").scrollHeight;
}
function status(t){$("status").textContent=t;}
function runtime(t,ok){$("runtime").textContent=t;$("runtimeDot").className="dot "+(ok?"ok":"warn");}

async function getDevice(){
  if(navigator.gpu){runtime("WebGPU available • GPU mode",true);return "webgpu";}
  runtime("WebGPU unavailable • CPU/WASM fallback",false);return "wasm";
}

async function loadModel(){
  var model=$("model").value.trim()||"onnx-community/Qwen2.5-Coder-1.5B-Instruct";
  if(generator&&activeModel===model)return;
  var device=await getDevice();
  status("Loading local AI model… first load downloads the model.");
  try{
    if(device==="webgpu"){
      try{
        generator=await pipeline("text-generation",model,{dtype:"q4f16",device:"webgpu"});
      }catch(e){
        runtime("WebGPU load failed • CPU/WASM fallback",false);
        generator=await pipeline("text-generation",model,{dtype:"q4"});
      }
    }else{
      generator=await pipeline("text-generation",model,{dtype:"q4"});
    }
    activeModel=model;
    status("Local AI ready.");
  }catch(e){
    generator=null;activeModel=null;
    throw new Error("Local model load failed: "+e.message);
  }
}

function messagesFor(prompt){
  var recent=chatHistory.slice(-6);
  var m=[{role:"system",content:SYSTEM}];
  recent.forEach(function(x){m.push(x);});
  m.push({role:"user",content:prompt});
  return m;
}

function outputText(result){
  if(!result||!result.length)return "";
  var g=result[0].generated_text;
  if(Array.isArray(g)){
    var last=g[g.length-1];
    return last&&last.content?String(last.content).trim():"";
  }
  return typeof g==="string"?g.trim():"";
}

async function ask(prompt){
  await loadModel();
  status("Generating locally on your device…");
  var result=await generator(messagesFor(prompt),{max_new_tokens:3000,do_sample:false,temperature:0.1,repetition_penalty:1.05});
  var out=outputText(result);
  if(!out)throw new Error("Local model returned an empty response.");
  chatHistory.push({role:"user",content:prompt},{role:"assistant",content:out});
  return out;
}

function cleanJson(text){
  var s=String(text||"").trim();
  var fence=String.fromCharCode(96,96,96);
  if(s.indexOf(fence)===0){
    var nl=s.indexOf("\n"),close=s.lastIndexOf(fence);
    if(nl>=0&&close>nl)s=s.slice(nl+1,close).trim();
  }
  var first=s.indexOf("{"),last=s.lastIndexOf("}");
  if(first>=0&&last>first)s=s.slice(first,last+1);
  return s;
}

function validateGraph(g){
  if(!g||!Array.isArray(g.nodes)||!Array.isArray(g.edges))throw new Error("Generated JSON must contain nodes and edges arrays.");
  var ids={};
  g.nodes.forEach(function(n){
    if(!n||!n.id)throw new Error("Every node must have an id.");
    if(ids[n.id])throw new Error("Duplicate node id: "+n.id);
    ids[n.id]=true;
  });
  g.edges.forEach(function(e){
    if(!e||!e.source||!e.target)throw new Error("Each edge needs source and target.");
    if(!ids[e.source])throw new Error("Edge source not found: "+e.source);
    if(!ids[e.target])throw new Error("Edge target not found: "+e.target);
  });
  if(!Array.isArray(g.validationNotes))g.validationNotes=[];
  g.validationNotes.push("Client-side JSON and node/edge reference validation passed.");
  return g;
}

function scorecardNeedsRepair(g,prompt){
  if(!/(scorecard|scoring|credit score|bre|vehicle loan)/i.test(prompt)) return false;
  var bad=false;
  function walk(x){
    if(bad||x===null||x===undefined)return;
    if(Array.isArray(x)){x.forEach(walk);return;}
    if(typeof x!=="object")return;
    var hasDecision=Object.prototype.hasOwnProperty.call(x,"decision")||Object.prototype.hasOwnProperty.call(x,"Decision");
    var decision=String(x.decision||x.Decision||"").toUpperCase();
    var value=x.extractedValue!==undefined?x.extractedValue:(x.value!==undefined?x.value:x["Extracted Value"]);
    var present=value!==undefined&&value!==null&&String(value).trim()!=="";
    if(hasDecision&&decision==="N/A"&&present){bad=true;return;}
    Object.keys(x).forEach(function(k){walk(x[k]);});
  }
  walk(g); return bad;
}

async function repairScorecard(g,prompt){
  var repair="Repair the current GoRules scorecard graph. The extracted values are already present. Every parameter with a non-empty value MUST have numeric score, maxScore and decision PASS/REFER/REJECT from explicit decisionTableNode scoring rules. N/A is permitted only for a genuinely missing value. A score of 0 may be used, but decision N/A is forbidden when a value exists. Ensure decision-table outputs are wired through expressions to parameterResults and aggregate totalScore, totalMaxScore, scorePercent, PASS/REFER/REJECT/N/A counts, finalDecision and riskCategory. Return only corrected graph JSON.\n\nCurrent graph:\n"+JSON.stringify(g)+"\n\nOriginal requirement:\n"+prompt;
  return parseGraph(await ask(repair));
}

function parseGraph(text){
  try{return validateGraph(JSON.parse(cleanJson(text)));}
  catch(e){throw new Error(e.message==="Unexpected end of JSON input"?"Incomplete JSON returned by the local model. Try a shorter prompt.":e.message);}
}

async function run(prompt){
  if(generating)return;
  add("user",prompt);
  generating=true;$("generate").disabled=true;$("send").disabled=true;
  try{
    var raw=await ask(prompt);
    add("bot",raw.slice(0,5000));
    lastGraph=parseGraph(raw);
    if(scorecardNeedsRepair(lastGraph,prompt)){
      status("Repairing scorecard outputs…");
      add("bot","The first graph left scored values as N/A. Running an automatic scoring-output repair…");
      lastGraph=await repairScorecard(lastGraph,prompt);
    }
    $("graph").textContent=JSON.stringify(lastGraph,null,2);
    $("notes").textContent=JSON.stringify(lastGraph.validationNotes||[],null,2);
    status("Graph generated successfully on your device.");
  }catch(e){
    add("bot","Error: "+e.message);
    status("Error.");
  }finally{
    generating=false;$("generate").disabled=false;$("send").disabled=false;
  }
}

$("generate").onclick=function(){
  var p=$("prompt").value.trim();if(p)run(p);
};
$("send").onclick=function(){
  var p=$("follow").value.trim();if(!p)return;
  $("follow").value="";
  run(lastGraph?"Current graph JSON:\n"+JSON.stringify(lastGraph)+"\n\nRevision request:\n"+p:p);
};
$("follow").addEventListener("keydown",function(e){if(e.key==="Enter"&&!e.shiftKey)$("send").click();});

$("clear").onclick=function(){
  chatHistory=[];lastGraph=null;$("messages").innerHTML="";
  $("graph").textContent="No graph generated yet.";
  $("notes").textContent="No validation notes yet.";
  status("Ready.");add("bot","Local AI builder ready. Describe the GoRules logic you want to build.");
};

$("sample1").onclick=function(){
  $("prompt").value="Create a collection date prediction graph using 6 months of Account Aggregator transactions. Use a configurable collectionThreshold, currently 15000. Identify up to 3 strong recurring collection dates. Keep business logic in ZEN Expression and Decision Table nodes. Use JavaScript only for external API fetching.";
};
$("sample2").onclick=function(){
  $("prompt").value="Create an NBFC BRE graph for applicant age, business vintage, loan amount, bureau score, FOIR and eligibility. Keep business logic in GoRules Expression and Decision Table nodes. Use JavaScript only for external API calls.";
};

$("tabGraph").onclick=function(){
  $("tabGraph").classList.add("active");$("tabNotes").classList.remove("active");
  $("graph").hidden=false;$("notes").hidden=true;
};
$("tabNotes").onclick=function(){
  $("tabNotes").classList.add("active");$("tabGraph").classList.remove("active");
  $("graph").hidden=true;$("notes").hidden=false;
};
$("download").onclick=function(){
  if(!lastGraph){alert("Generate a graph first.");return;}
  var blob=new Blob([JSON.stringify(lastGraph,null,2)],{type:"application/json"});
  var u=URL.createObjectURL(blob),a=document.createElement("a");
  a.href=u;a.download="gorules-local-graph.json";document.body.appendChild(a);a.click();a.remove();
  setTimeout(function(){URL.revokeObjectURL(u);},1000);
};
$("model").addEventListener("change",function(){
  generator=null;activeModel=null;status("Model changed. It will load on the next generation.");
});

getDevice().then(function(){
  add("bot","Ready. No API key and no cloud AI credits are required. Inference runs locally in your browser.");
});
