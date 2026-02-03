'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';

interface Actor {
  actor_id: string;
  name: string;
  thumbnail: string;
  description: string;
}

interface WordTimestamp {
  text: string;
  start: number;
  end: number;
}

interface Sentence {
  text: string;
  start: number;
  end: number;
  words: WordTimestamp[];
}

interface Video {
  video_id: string;
  actor_id: string;
  title: string;
  video_url: string;
  thumbnail?: string;
  script?: string;  // 영상별 대사
  sentences?: Sentence[];  // 문장 단위 + 단어별 타임스탬프
}

interface VideoThumbnailProps {
  src: string;
  className?: string;
  title: string;
}

function VideoThumbnail({ src, className, title }: VideoThumbnailProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const handleLoadedMetadata = () => {
    if (!videoRef.current) {
      return;
    }
    // 첫 프레임을 안정적으로 보여주기 위해 아주 살짝 이동
    videoRef.current.currentTime = 0.01;
  };

  const handleSeeked = () => {
    if (!videoRef.current) {
      return;
    }
    videoRef.current.pause();
  };

  return (
    <video
      ref={videoRef}
      src={src}
      preload="metadata"
      muted
      playsInline
      className={className}
      aria-label={title}
      onLoadedMetadata={handleLoadedMetadata}
      onSeeked={handleSeeked}
    />
  );
}

export default function RecommendPage() {
  const router = useRouter();
  
  const [loading, setLoading] = useState(true);
  const [userNickname, setUserNickname] = useState('');
  const [recommendedActor, setRecommendedActor] = useState<Actor | null>(null);
  const [myActorVideos, setMyActorVideos] = useState<Video[]>([]);
  const [allVideos, setAllVideos] = useState<Video[]>([]);
  const [matchRate, setMatchRate] = useState(0); 

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (!storedUser) {
      alert('로그인이 필요합니다.');
      router.push('/');
      return;
    }

    const userData = JSON.parse(storedUser);
    setUserNickname(userData.nickname);
    const actorId = userData.recommended_actor_id;

    if (!actorId) {
      setLoading(false);
      return;
    }

    if (typeof userData.recommended_actor_score === 'number') {
      const raw = Math.max(0, Math.min(1, userData.recommended_actor_score));
      setMatchRate(Math.round(raw * 100));
    } else {
      setMatchRate(Math.floor(Math.random() * (98 - 85 + 1)) + 85);
    }

    const fetchData = async () => {
      try {
        const actorsRes = await fetch('http://127.0.0.1:8000/api/recommend/actors');
        const actorsData: Actor[] = await actorsRes.json();
        const myActor = actorsData.find(a => a.actor_id === actorId);
        setRecommendedActor(myActor || null);

        const myVideosRes = await fetch(`http://127.0.0.1:8000/api/actors/${actorId}/videos`);
        if (myVideosRes.ok) {
          setMyActorVideos(await myVideosRes.json());
        }

        const allVideosRes = await fetch('http://127.0.0.1:8000/api/recommend/videos');
        if (allVideosRes.ok) {
          setAllVideos(await allVideosRes.json());
        }

      } catch (err) {
        console.error("데이터 로딩 실패:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [router]);

  const handleSelectVideo = (video: Video) => {
    localStorage.setItem('selected_video', JSON.stringify(video));
    router.push('/acting');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mr-3"></div>
        분석 결과를 불러오는 중...
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-[#0f0f0f] text-white p-6 pb-20 overflow-y-auto">
      
      {/* ──────────────────────────────────────────────
          상단 섹션: (좌) 배우 카드 + (우) 추천 명장면
      ────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto mt-8 mb-16">
        <div className="flex flex-col lg:flex-row gap-8 items-start">
          
          {/* [좌측] 내 닮은꼴 배우 카드 (디자인 수정됨) */}
          <div className="w-full lg:w-1/3 flex-shrink-0 flex flex-col items-center lg:items-start">
            <h1 className="text-2xl font-bold mb-6 text-center lg:text-left w-full">
              <span className="text-blue-400">{userNickname}</span>님의 닮은꼴
            </h1>
            
            {recommendedActor ? (
              // ▼▼▼ 여기가 수정된 디자인 (이전 스타일 복원 + 크기 축소) ▼▼▼
              <div className="bg-gray-800/80 backdrop-blur-md rounded-3xl p-6 shadow-2xl border border-gray-700 w-full max-w-[320px] transform transition hover:scale-105 duration-300">
                <div className="relative w-40 h-40 mx-auto mb-4 rounded-full overflow-hidden border-4 border-blue-500 shadow-lg">
                  <img 
                    src={`http://127.0.0.1:8000${recommendedActor.thumbnail}`} 
                    alt={recommendedActor.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="text-center">
                  <h2 className="text-2xl font-bold mb-2">{recommendedActor.name}</h2>
                  <div className="inline-block px-4 py-1.5 bg-blue-600/20 text-blue-400 rounded-full text-sm font-bold mb-4 border border-blue-500/30">
                    싱크로율 {matchRate}%
                  </div>
                  <p className="text-gray-400 text-sm leading-relaxed border-t border-gray-700 pt-4">
                    {recommendedActor.description}
                  </p>
                </div>
              </div>
            ) : (
              <div className="bg-gray-800 p-6 rounded-xl text-center text-gray-400 w-full max-w-[320px]">
                배우 정보를 불러오지 못했습니다.
              </div>
            )}
          </div>

          {/* [우측] 추천 명장면 따라하기 (가로 스크롤) */}
          <div className="w-full lg:w-2/3 pt-2">
            <div className="flex items-center gap-2 mb-4">
              <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
              </svg>
              <h3 className="text-xl font-bold">이 배우의 명장면 따라하기</h3>
            </div>

            <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide snap-x">
              {myActorVideos.length > 0 ? (
                myActorVideos.map((video) => (
                  <div 
                    key={video.video_id}
                    onClick={() => handleSelectVideo(video)}
                    className="flex-none w-64 cursor-pointer group snap-start"
                  >
                    <div className="relative aspect-video bg-gray-800 rounded-lg overflow-hidden border border-gray-700 mb-2 shadow-md group-hover:shadow-blue-500/20 transition-all">
                      <VideoThumbnail
                        src={`http://127.0.0.1:8000${video.video_url}`}
                        title={video.title}
                        className="w-full h-full object-cover opacity-80 group-hover:scale-105 transition-transform duration-500"
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/10 transition-colors">
                        <div className="w-10 h-10 rounded-full bg-black/60 flex items-center justify-center backdrop-blur-sm group-hover:scale-110 transition-transform">
                           <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                        </div>
                      </div>
                    </div>
                    <h4 className="font-medium text-white group-hover:text-blue-400 truncate transition-colors">{video.title}</h4>
                  </div>
                ))
              ) : (
                <div className="w-full h-40 flex items-center justify-center bg-[#1e1e1e] rounded-xl text-gray-500 border border-dashed border-gray-700">
                  추천 영상이 없습니다 😢
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <hr className="border-gray-800 max-w-7xl mx-auto my-12" />

      {/* ──────────────────────────────────────────────
          하단 섹션: 유튜브 스타일 전체 영상 그리드
      ────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <span>🎬</span> 모든 연기 영상 탐색
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-x-4 gap-y-8">
          {allVideos.map((video) => (
            <div 
              key={video.video_id} 
              onClick={() => handleSelectVideo(video)}
              className="cursor-pointer group"
            >
              <div className="relative aspect-video bg-gray-800 rounded-xl overflow-hidden mb-3 border border-transparent group-hover:border-gray-600 transition-all shadow-sm group-hover:shadow-lg">
                <VideoThumbnail
                  src={`http://127.0.0.1:8000${video.video_url}`}
                  title={video.title}
                  className="w-full h-full object-cover"
                />
                <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center text-gray-500 hidden">
                  <svg className="w-12 h-12 opacity-50" fill="currentColor" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
                </div>
                
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                   <svg className="w-12 h-12 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="flex-col overflow-hidden">
                  <h4 className="text-white font-semibold text-sm leading-tight mb-1 line-clamp-2 group-hover:text-blue-400 transition-colors">
                    {video.title}
                  </h4>
                  <p className="text-gray-400 text-xs">
                    {video.actor_id}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

    </main>
  );
}