import {memo,useEffect,useState} from 'react';

export const DetailHint=memo(function DetailHint({saver=false,mobile=false}){
  const [visible,setVisible]=useState(true);
  useEffect(()=>{
    const timer=setTimeout(()=>setVisible(false),8500);
    return()=>clearTimeout(timer);
  },[]);
  if(!visible)return null;
  return <div className="detail-hint" role="status" aria-live="polite">
    <div className="detail-hint-content"><strong>{saver?'Data saver is on':'Details load up close'}</strong><span>{saver?(mobile?'Explore with lighter detail. Open on a computer for higher visual quality.':'Explore with lighter detail. Select web preview for more detail.'):'Move closer and pause briefly for available building detail and textures to load.'}</span></div>
  </div>;
});
