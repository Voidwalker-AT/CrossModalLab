import {useEffect, useRef, useState} from 'react';
import type {ReactNode, KeyboardEvent} from 'react';

/** Keep one copy of the figure and its controls, including SVG marker IDs. */
export function Enlarge({title,children}:{title:string;children:ReactNode}) {
 const [expanded,setExpanded]=useState(false);
 const frame=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null),close=useRef<HTMLButtonElement>(null);
 useEffect(()=>{
  if(!expanded)return;
  const overflow=document.body.style.overflow;
  document.body.style.overflow='hidden';close.current?.focus();
  return ()=>{document.body.style.overflow=overflow;trigger.current?.focus()};
 },[expanded]);
 function keyboard(event:KeyboardEvent){
  if(!expanded)return;
  if(event.key==='Escape'){event.preventDefault();event.stopPropagation();setExpanded(false)}
  if(event.key==='Tab'){
   const controls=Array.from(frame.current?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href],input,select,summary,[tabindex="0"]')??[]).filter(el=>el.checkVisibility());
   const first=controls[0],last=controls[controls.length-1];
   if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}
   else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}
  }
 }
 return <div ref={frame} className={`enlarge-frame${expanded?' expanded':''}`} role={expanded?'dialog':undefined} aria-modal={expanded?true:undefined} aria-label={expanded?title:undefined} onKeyDown={keyboard}>
  <div className="enlarge-surface">
   <div className="enlarge-toolbar">{expanded?<><strong>{title}</strong><button ref={close} className="button ghost" onClick={()=>setExpanded(false)}>Close enlarged view</button></>:<button ref={trigger} className="button ghost" onClick={()=>setExpanded(true)}>Enlarge {title}</button>}</div>
   {expanded&&<p className="caption">Scroll within this view to inspect the full-size figure. Escape closes it.</p>}
   <div className="enlarge-scroll"><div className="enlarge-content">{children}</div></div>
  </div>
 </div>
}
