"""Readable scientific figure variants; preserve the frozen analyzer's original PNGs."""
import math
from pathlib import Path
import sys
import numpy as np
from reasoning_rl.runtime import ROOT,json_write
sys.path.insert(0,str(ROOT/"scripts"))
from filter_common import FAMILY,read


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=ROOT/"reports/comparisons"/FAMILY
    summary=read(out/"summary.json")
    rows=summary["endpoints"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout="constrained")
    for j,(seed,marker,style,color) in enumerate(zip((13,29,47),("o","s","^"),("-","--",":"),("#27638B","#9C6525","#7860A4"),strict=True)):
        pair=[next(r for r in rows if r["seed"]==seed and r["arm"]==arm) for arm in ("U","F")]
        values=[r["accuracy"]*100 for r in pair]
        xs=[(j-1)*.08,1+(j-1)*.08]
        ax.plot(xs,values,marker=marker,ls=style,color=color,lw=1.8,ms=8,label=f"seed {seed}")
        for x,y,r in zip(xs,values,pair,strict=True):
            ax.annotate(f"{r['correct']}/128",(x,y),xytext=(8,(j-1)*13+4),textcoords="offset points",fontsize=10)
    baseline=summary["s2_baseline"]["accuracy"]*100
    ax.axhline(baseline,color="#666666",ls="-.",lw=1.1,label=f"fixed S2: {summary['s2_baseline']['correct']}/128")
    ys=[r["accuracy"]*100 for r in rows]+[baseline]
    ax.set(xticks=[0,1],xticklabels=["U: uniform on 64 candidates","F: uniform on 16 retained"],
        ylabel="Development accuracy (%)",xlim=(-.35,1.5),ylim=(math.floor(min(ys))-2,math.ceil(max(ys))+3),
        title="All paired endpoints at the frozen common budget\nReused development128; fresh-process reload; formal tests sealed")
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="upper left",ncol=2,fontsize=9)
    fig.savefig(out/"accuracy_paired_accessible.png",dpi=170)
    plt.close(fig)
    fig=plt.figure(figsize=(9.5,8.5),layout="constrained")
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.05])
    a0=fig.add_subplot(grid[0,0]); a1=fig.add_subplot(grid[0,1]); a2=fig.add_subplot(grid[1,:])
    arms=summary["by_arm"]
    bottom=np.zeros(2)
    for key,label,color in (("all_correct_groups","All correct","#9BB5CA"),("all_wrong_groups","All wrong","#D5D9DF"),("mixed_groups","Mixed rewards","#D48D36")):
        values=np.array([a[key]/a["groups"]*100 for a in arms])
        a0.bar([0,1],values,bottom=bottom,color=color,width=.6,label=label,edgecolor="white",lw=.7)
        for x,v,b in zip((0,1),values,bottom,strict=True):
            if v>4:
                a0.text(x,b+v/2,f"{v:.1f}%",ha="center",va="center",fontsize=10)
        bottom+=values
    a0.set(xticks=[0,1],xticklabels=["U","F"],ylabel="Actual training rollout groups (%)",ylim=(0,100),title="Reward variation during training")
    a0.legend(loc="upper center",bbox_to_anchor=(.5,-.08),ncol=1,fontsize=9)
    rates=[a["effective_completions_per_paid_end_to_end_second"] for a in arms]
    a1.bar([0,1],rates,color=["#2874A6","#14866D"],width=.6)
    for x,v in enumerate(rates):
        a1.text(x,v,f"{v:.4f}",ha="center",va="bottom")
    a1.set(xticks=[0,1],xticklabels=["U","F"],ylabel="Effective advantage completions / second",
        ylim=(0,max(rates)*1.25),title="Paid signal efficiency\nIncludes full probe and reload evaluation")
    bottom=np.zeros(2)
    for field,label,color in (("training_process_seconds","Training process","#729DBB"),("probe_standalone_debit_seconds","Full probe debit","#D48D36"),("evaluation_process_seconds","Reload evaluation","#75AD9A"),("parent_bookkeeping_seconds","Parent bookkeeping","#B7B5B4")):
        values=np.array([a[field]/3 for a in arms])
        a2.bar([0,1],values,bottom=bottom,color=color,width=.55,label=label)
        bottom+=values
    for x,v in enumerate(bottom):
        a2.text(x,v+12,f"{v:.1f} s",ha="center",va="bottom")
    cap=rows[0]["common_cap_seconds"]
    a2.axhline(cap,color="#333333",ls="--",lw=1)
    a2.text(1.35,cap,f"Common cap: {cap:.0f} s",va="bottom",ha="right",fontsize=10)
    a2.set(xticks=[0,1],xticklabels=["U","F"],ylabel="Mean charged endpoint seconds",ylim=(0,cap*1.1),xlim=(-.6,1.5),
        title="Nonoverlapping charged endpoint costs; three-seed means")
    a2.legend(loc="upper center",bbox_to_anchor=(.5,-.08),ncol=2,fontsize=9)
    fig.savefig(out/"mechanism_and_cost_accessible.png",dpi=170)
    plt.close(fig)
    json_write(out/"figure_variants.json",{"source":"summary.json","original_frozen_analyzer_images_preserved":True,
        "changes":"Distinct seed marker shapes and line styles; two-row layout for legible labels; no metric or endpoint changes",
        "images":["accuracy_paired_accessible.png","mechanism_and_cost_accessible.png"]})


if __name__=="__main__":
    main()
