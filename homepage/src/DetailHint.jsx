import {memo,useEffect,useState} from 'react';

export const DetailHint=memo(function DetailHint(){
  const [visible,setVisible]=useState(true);
  useEffect(()=>{
    const timer=setTimeout(()=>setVisible(false),8500);
    return()=>clearTimeout(timer);
  },[]);
  if(!visible)return null;
  return <div className="detail-hint" role="status" aria-live="polite">
    <div className="detail-hint-content"><strong>Details load up close</strong><span>Move closer and pause briefly for higher-detail models and textures to load.</span></div>
  </div>;
});
